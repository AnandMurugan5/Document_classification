import streamlit as st
import os
from email_extraction import EmailData
from model_training import ContentPrediction,DocumentPrediction

def login(email, password):
    try:
        email_client = EmailData(email, password)
        email_client.login()
        return email_client
    except Exception as e:
        st.warning('Please login to access the page')
        return None
st.title('Email classification - Documents and Content')
dashboard = st.sidebar

user_id = dashboard.text_input(label='User Id:', placeholder='Enter the email id...')
user_password = dashboard.text_input(label='Password:', type='password', placeholder='Enter the secret code...')

dashboard.button('Login')
email_client = login(user_id, user_password)
if email_client:
    st.success('Login successful!')
operation = dashboard.selectbox('Select Operation:', ['Select...', 'Download Attachments', 'Content'], index=0)
if operation == 'Download Attachments':
    email_attachments = email_client.download_attachments()
    email_attachments_pdf = [os.path.join(root, filename) for root, dirs, files in os.walk('Documents/') for filename in files if filename.endswith('.pdf')]
    documents = DocumentPrediction(email_attachments_pdf)
    result = documents.processing()
    dashboard.success("Document Processing Complete!")
    st.dataframe(result)
    pass

elif operation == 'Content':
    email_content = email_client.fetch_emails()
    content = ContentPrediction(content=email_content)
    result = content.processing()
    dashboard.success("Content Processing Complete!")
    st.dataframe(result)
    pass
        
if dashboard.button('Logout'):
    file_path = "Documents/"  
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"File {file_path} deleted successfully!")
        email_client.logout()
    else:
        st.warning(f"File {file_path} not found!")
        
    
# else:
#     st.warning('Please login to access the page')
