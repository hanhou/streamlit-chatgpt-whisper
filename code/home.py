import streamlit as st
import openai
import wave
from gtts import gTTS
import base64
from dataclasses import dataclass, field
import time
import pandas as pd

import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import streamlit_nested_layout
import pykakasi
from audio_recorder_streamlit import audio_recorder
from streamlit_chat import message


st.set_page_config(layout="wide")

kks = pykakasi.kakasi()

# Set the model engine and your OpenAI API key
openai.api_key = os.getenv("API_KEY")



@st.cache_data(max_entries=100)
def get_transcription(file, lang=['ja'], filelen=None):
    with open(file, "rb") as f:
        try:
            transcription = openai.Audio.transcribe("whisper-1", f, language=lang).text
        except:
            transcription = None
    return transcription


@st.cache_data(max_entries=100)
def ChatGPT(user_query, model="gpt-3.5-turbo"):
    ''' 
    This function uses the OpenAI API to generate a response to the given 
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user",
                   "content": user_query}]
                                      )
    completion.choices[0].message.content = completion.choices[0].message.content.lstrip('ChatGPT').lstrip('chatGPT').lstrip(':').lstrip(' ').lstrip('\n')
    st.session_state.total_tokens += completion.usage.total_tokens
    return completion
    
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
        
def show_markdown(text, color='black', font_size='15', align='left', col=None, other_styles=''):
    c = st if col is None else col
    c.markdown(rf"<span style='font-size:{font_size}px; color:{color}; text-align:{align} ; {other_styles}'>{text}</span>", unsafe_allow_html=True)

        
def show_kakasi(text, hira_font=15, orig_font=25, n_max_char_each_row=50, col=None):
    text = kks.convert(text)

    # Interpret string
    len_each_section = [max(len(this['orig']), len(this['hira'])) for this in text]
    cumulative_lens = np.cumsum(len_each_section)
    start_at_char = np.arange(np.ceil(max(cumulative_lens) / n_max_char_each_row )) * n_max_char_each_row
    which_section_to_start = list(np.searchsorted(cumulative_lens, start_at_char))
    which_section_to_start.append(len(text) + 1)
    
    if col is None:
        col = st.columns([1])[0]
    
    with col:
        for start, end in zip(which_section_to_start[:-1], which_section_to_start[1:]):
            this_row = text[start:end]
            if not this_row:
                continue
            cols = st.columns(len_each_section[start:end])
            for i, section in enumerate(this_row):
                if section['hira'] != section['orig']:
                    show_markdown(section['hira'], color='blue', font_size=hira_font, col=cols[i])
                else:
                    show_markdown('<br>', color='blue', font_size=hira_font, col=cols[i])
                    
                show_markdown(section['orig'], font_size=orig_font, col=cols[i])
                
@dataclass
class Chat:
    qa: pd.DataFrame
    context: list
    length: int = 0
    
    def __init__(self):
        self.qa = pd.DataFrame(columns=['Q', 'A', 'Q_zh', 'A_zh'])
        self.context = background_promt
        
    def _gen_setting(self, field, container=None):
        if field == 'Q':
            return dict(font_size=20, col=self.col_dialog.columns([2, 1])[0] if container is None else container)
        if field == 'Q_zh':
            return dict(font_size=15, color='blue', col=self.col_dialog.columns([2, 1])[0] if container is None else container)
        if field == 'A':
            return dict(font_size=20, color='black', col=self.col_dialog.columns([1, 2])[1] if container is None else container, other_styles='font-weight:bold')
        if field == 'A_zh':
            return dict(font_size=15, color='blue', col=self.col_dialog.columns([1, 2])[1] if container is None else container   , other_styles='font-weight:bold')
    
    def add(self, field, text, **kwarg):
        if field == 'Q':
            self.length += 1
            
        self.qa.loc[self.length, field] = text
        
        show_markdown(text, **self._gen_setting(field, **kwarg))
        

    def show_history(self):
        if not len(self.qa): return
        
        for row in self.qa[::-1][1:].itertuples():
            show_markdown(row.A, **self._gen_setting('A'))
            if st.session_state.if_chinese:
                if pd.isna(row.A_zh):
                    response = ChatGPT('Translate to Simplified Chinese:' + row.A)
                    self.qa.loc[row.Index, 'A_zh'] = response.choices[0].message.content

                show_markdown(self.qa.loc[row.Index, 'A_zh'], **self._gen_setting('A_zh'))

            show_markdown(row.Q, **self._gen_setting('Q'))
            if st.session_state.if_chinese:
                
                if pd.isna(row.Q_zh):
                    response = ChatGPT('Translate to Simplified Chinese:' + row.Q)
                    self.qa.loc[row.Index, 'Q_zh'] = response.choices[0].message.content

                show_markdown(self.qa.loc[row.Index, 'Q_zh'], **self._gen_setting('Q_zh'))
                

    def generate_query(self):
        query = self.context
        
        for row in self.qa.itertuples():
            query += f'Yixi: {row.Q}\n'
            if not pd.isna(row.A): 
                query += f'chatGPT: {row.A}\n'
        
        st.sidebar.write(query)
        return query    
    
def init():
    st.session_state.chat = Chat()
    st.session_state.total_tokens = 0
    st.session_state.text_query = ''
    

def clear_text():
    st.session_state.text_query = st.session_state.text
    st.session_state.text = ""

# st.title("Chatting with ChatGPT")
with st.sidebar:
    st.title("跟着chatGPT学日语")
    
    audio_bytes = audio_recorder(
                             pause_threshold=2.0, 
                             sample_rate=44100,
                             text="",
                             recording_color="E9E9E9",
                             neutral_color="575757",
                             icon_name="paw",
                             icon_size="6x",
                             )
        
    h_replay = st.container()
    
    if st.button('Clear conversation'):
        init()
        st.experimental_rerun()
            
    st.text_input("Start chat here :q", "", key="text", on_change=clear_text)
    
    st.session_state.if_chinese = st.checkbox('Show Chinese', True)
    if_hira = st.checkbox('Show Hira', True)
    if_slow = st.checkbox('Read it slow', False)
        
    for i in range(30):
        st.write('\n')
    st.markdown('---')
    st.markdown("🎂希希希生日快乐！！！🎂")
    st.markdown("Designed by Han with ❤️ @ 2023.3")

    
if 'chat' not in st.session_state:
    init()
    
st.sidebar.write(st.session_state.total_tokens)    


col_dialog, _, col_hira, _ = st.columns([2, 0.2, 1, 0.5])
st.session_state.chat.col_dialog = col_dialog
st.session_state.chat.col_hira = col_hira


if audio_bytes:
    h_replay.audio(audio_bytes, format="audio/wav")
    
    # Create a WAV file object
    wav_file = wave.open('output.wav', 'w')
    wav_file.setparams((2, 2, 44100, len(audio_bytes), 'NONE', 'not compressed'))
    wav_file.writeframes(audio_bytes)
    wav_file.close()
    user_query = get_transcription('output.wav', lang=['ja'], filelen=os.path.getsize('output.wav')) # If size changes, redo transcribe
    

if st.session_state.text_query != ":q" or st.session_state.text_query != "":
    user_query = st.session_state.text_query
    
    
if user_query is not None and user_query != "":
    st.session_state.text_query = ''
    
    container_answer = col_dialog.columns([1, 2])[1]
    
    st.session_state.chat.add('Q', user_query)
    
    if st.session_state.if_chinese:
        response = ChatGPT('Translate to Simplified Chinese:' + user_query)
        st.session_state.chat.add('Q_zh', response.choices[0].message.content)
    
    query = st.session_state.chat.generate_query()
    
    response = ChatGPT(query)
    response_text = response.choices[0].message.content
    st.session_state.chat.add('A', response_text, container=container_answer)
    
    if st.session_state.if_chinese:        
        response = ChatGPT('Translate to Simplified Chinese:' + response_text)
        st.session_state.chat.add('A_zh', response.choices[0].message.content, container=container_answer)
        
    st.session_state.chat.show_history()

    if if_hira:
        show_kakasi(response_text, hira_font=15, orig_font=25, n_max_char_each_row=20, col=col_hira)


    # Create a TTS object
    tts = gTTS(response_text, lang='ja', slow=if_slow)
    tts.save('response.mp3')
    autoplay_audio('response.mp3')
    

    st.audio('response.mp3')
    
st.session_state.chat.qa
