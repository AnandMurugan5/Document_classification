import streamlit as st
import os
from email_extraction import EmailData
from model_training import DocumentPrediction,ContentPrediction
# Function to login and store session state
def login(email, password):
    try:
        email_client = EmailData(email, password)
        email_client.login()
        st.session_state.email_client = email_client
        st.session_state.logged_in = True
        return True
    except Exception as e:
        st.warning('Please login to access the page')
        return False

# Function to handle logout and delete files
def logout():
    if hasattr(st.session_state, 'email_client') and st.session_state.logged_in:
        # Delete files in the "Documents" directory if it exists
        documents_dir = 'Documents/'
        if os.path.exists(documents_dir) and os.path.isdir(documents_dir):
            for file_name in os.listdir(documents_dir):
                file_path = os.path.join(documents_dir, file_name)
                try:
                    os.remove(file_path)
                    st.success(f"File {file_path} deleted successfully!")
                except Exception as e:
                    st.warning(f"Failed to delete file {file_path}. Error: {e}")

        # Logout
        st.session_state.email_client.logout()
        st.success('Logout successful!')
        st.session_state.logged_in = False

# Initialize session state
if 'email_client' not in st.session_state:
    st.session_state.email_client = None
    st.session_state.logged_in = False

st.title('Email classification - Documents and Content')
dashboard = st.sidebar

user_id = dashboard.text_input(label='User Id:', placeholder='Enter the email id...')
user_password = dashboard.text_input(label='Password:', type='password', placeholder='Enter the secret code...')

if dashboard.button('Login'):
    if login(user_id, user_password):
        st.success('Login successful!')

operation = dashboard.selectbox('Select Operation:', ['Select...', 'Download Attachments', 'Content'], index=0)

if operation == 'Download Attachments':
    if st.session_state.logged_in:
        email_attachments = st.session_state.email_client.download_attachments()
        email_attachments_pdf = [os.path.join(root, filename) for root, dirs, files in os.walk('Documents/') for filename in files if filename.endswith('.pdf')]
        documents = DocumentPrediction(email_attachments_pdf)
        result = documents.processing()
        dashboard.success("Document Processing Complete!")
        st.dataframe(result)
    else:
        st.warning('Please login to access')


elif operation == 'Content':
    if st.session_state.logged_in:
        email_content = st.session_state.email_client.fetch_emails()
        content = ContentPrediction(content=email_content)
        result = content.processing()
        dashboard.success("Content Processing Complete!")
        st.dataframe(result)
    else:
        st.warning('Please login to access')

if dashboard.button('Logout'):
    logout()
