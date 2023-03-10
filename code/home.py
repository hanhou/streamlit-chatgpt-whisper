import streamlit as st
import openai
import wave
from gtts import gTTS
import base64

import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import streamlit_nested_layout
import pykakasi
from audio_recorder_streamlit import audio_recorder


st.set_page_config(layout="wide")

kks = pykakasi.kakasi()

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
    
    if_chinese = st.checkbox('Show Chinese', True)
    if_hira = st.checkbox('Show Hira', True)
    if_slow = st.checkbox('Read it slow', False)
    
    for i in range(30):
        st.write('\n')
    st.markdown('---')
    st.markdown("🎂希希希生日快乐！！！🎂")
    st.markdown("Designed by Han with ❤️ @ 2023.3")



# Set the model engine and your OpenAI API key
openai.api_key = os.getenv("API_KEY")

background_promt = '''I am Miss YiXi. I am learning Japanese. Please chat with me in simple Japanese.
                      By the way, my dog's name is rice pudding. Try not to exceed 30 words for each answer.
                      \n'''

def main():
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("Enter query here, to exit enter :q", "what is Python?")
    if user_query != ":q" or user_query != "":
        # Pass the query to the ChatGPT function
        response = ChatGPT(user_query)
        st.write(f'{response.choices[0].message.content}')
        st.write(f"total tokens = {response.usage.total_tokens}")
        return


@st.cache_data(max_entries=20)
def get_transcription(file, lang=['ja'], filelen=None):
    with open(file, "rb") as f:
        try:
            transcription = openai.Audio.transcribe("whisper-1", f, language=lang).text
        except:
            transcription = None
    return transcription


@st.cache_data(max_entries=20)
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
    text = kks.convert(response_text)

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


if audio_bytes:
    h_replay.audio(audio_bytes, format="audio/wav")
    
    # Create a WAV file object
    wav_file = wave.open('output.wav', 'w')
    wav_file.setparams((2, 2, 44100, len(audio_bytes), 'NONE', 'not compressed'))
    wav_file.writeframes(audio_bytes)
    wav_file.close()
    
#if 1:
    
    user_query = get_transcription('output.wav', lang=['ja'], filelen=os.path.getsize('output.wav')) # If size changes, redo transcribe
    
    col_dialog, _, col_hira, _ = st.columns([1, 0.2, 1, 0.5])
    
    if user_query is not None:
        show_markdown(user_query, font_size=20, col=col_dialog.columns([2, 1])[0])
                
        if if_chinese:
            query_to_Chinese = ChatGPT('Translate to Simplified Chinese:' + user_query).choices[0].message.content
            show_markdown(query_to_Chinese, font_size=15, color='blue', col=col_dialog.columns([2, 1])[0])
        
        response = ChatGPT(background_promt + user_query)
        response_text = response.choices[0].message.content
        show_markdown(response_text, font_size=20, color='black', col=col_dialog.columns([1, 2])[1], other_styles='font-weight:bold')

        if if_hira:
            show_kakasi(response_text, hira_font=15, orig_font=25, n_max_char_each_row=20, col=col_hira)

        if if_chinese:
            response_to_Chinese = ChatGPT('Translate to Simplified Chinese.:' + response_text).choices[0].message.content
            show_markdown(response_to_Chinese, font_size=15, color='blue', col=col_dialog.columns([1, 2])[1], other_styles='font-weight:bold')

        # Create a TTS object
        tts = gTTS(response_text, lang='ja', slow=if_slow)
        tts.save('response.mp3')
        autoplay_audio('response.mp3')
        
        
        st.audio('response.mp3')
