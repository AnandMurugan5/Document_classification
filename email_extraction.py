import imaplib
import email
import os

class EmailData:
    def __init__(self, userid, password) -> None:
        self.userid = userid
        self.password = password
        self.mail = None

    def login(self):
        self.mail = imaplib.IMAP4_SSL('imap.gmail.com')
        self.mail.login(self.userid, self.password)

    def logout(self):
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
            except imaplib.IMAP4.error as e:
                print(f"Error during logout: {e}")
            finally:
                self.mail = None

    def raw_emails(self):
        self.login()
        self.mail.select('inbox')
        print(f"Login: {self.userid}")
        _, search_data = self.mail.search(None, 'ALL')
        mail_ids = search_data[0].split()
        raw_emails = []
        for mail_id in mail_ids:
            _, data = self.mail.fetch(mail_id, '(RFC822)')
            raw_email = data[0][1]
            raw_emails.append(raw_email)
        return raw_emails

    def fetch_emails(self):
        emails = []
        try:
            raw_emails = self.raw_emails()
            print("___Fatching Emails___")
            for raw_email in raw_emails:
                email_message = email.message_from_bytes(raw_email)
                subject = email_message['subject']
                body = ""

                if email_message.is_multipart():
                    for part in email_message.walk():
                        part_content_type = part.get_content_type()
                        if part_content_type == "text/plain":
                            body = part.get_payload(decode=True)
                            if isinstance(body, bytes):
                                body = body.decode('utf-8', 'ignore')
                else:
                    body = email_message.get_payload(decode=True)
                    if isinstance(body, bytes):
                        body = body.decode('utf-8', 'ignore')

                emails.append((subject, body))

        except Exception as e:
            print(f"Error occurred while processing email: {e}")

        return emails

    def download_attachments(self):
        DOWNLOAD_DIR = './Documents'
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        try:
            print("___Downloadind started___")
            raw_emails = self.raw_emails()
            for raw_email in raw_emails:
                msg = email.message_from_bytes(raw_email)

                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue

                    filename = part.get_filename()
                    if filename:
                        filepath = os.path.join(DOWNLOAD_DIR, filename)
                        with open(filepath, 'wb') as f:
                            f.write(part.get_payload(decode=True))

            self.logout()

        except Exception as e:
            print(f"Error downloading attachments: {e}")


# email_client = EmailData('anandraman249@gmail.com', 'ffmz aqom nbhb bldk')
# email_client.login()
# emails_data = email_client.download_attachments()