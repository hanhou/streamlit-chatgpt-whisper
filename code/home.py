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
from st_custom_components import st_audiorec


# st.title("Chatting with ChatGPT")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with 
       the OpenAI API's implementation of the ChatGPT model.
       Enter a **query** in the **text box** and **press enter** to receive 
       a **response** from the ChatGPT
       '''
    )

# Set the model engine and your OpenAI API key
openai.api_key = os.getenv("API_KEY")


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

# main()

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # Create a WAV file object
    wav_file = wave.open('output.wav', 'w')
    wav_file.setparams((2, 2, 44100, len(wav_audio_data), 'NONE', 'not compressed'))

    # Write the audio data to the WAV file
    wav_file.writeframes(wav_audio_data)

    # Close the WAV file
    wav_file.close()

    with open('output.wav', "rb") as f:
        transcription = openai.Audio.transcribe("whisper-1", f, language=['en', 'ja'])#, 'zh'])

    user_query = transcription.text
    st.write(user_query)
    
    response_to_Chinese = ChatGPT('Translate from Japanese into Chinese:' + user_query)
    st.write(f'{response_to_Chinese.choices[0].message.content}')


    response = ChatGPT(user_query)
    response_text = response.choices[0].message.content
    st.write(f'{response_text}')
    
    response_to_Chinese = ChatGPT('Translate from Japanese into Chinese:' + response_text)
    st.write(f'{response_to_Chinese.choices[0].message.content}')


    # Create a TTS object
    tts = gTTS(response_text, lang='ja', slow=False)

    # Save the audio file
    tts.save('response.mp3')

    # Play the audio file
    autoplay_audio('response.mp3')


