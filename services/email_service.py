import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_application_confirmation_email(recipient_email: str, user_name: str, job_title: str, company_name: str):
    """
    Sends a job application confirmation email with a WhatsApp button to contact HR.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from_email = os.getenv("SMTP_FROM_EMAIL")
    smtp_from_name = os.getenv("SMTP_FROM_NAME", "HR Team")

    if not all([smtp_host, smtp_user, smtp_password, smtp_from_email]):
        print("SMTP configuration is missing. Email not sent.")
        return False

    subject = f"Application Received: {job_title} at {company_name}"
    
    # WhatsApp link
    whatsapp_number = "917304879310"
    whatsapp_message = f"Hello HR, I have applied for the {job_title} position."
    whatsapp_link = f"https://wa.me/{whatsapp_number}?text={whatsapp_message.replace(' ', '%20')}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .button {{
                background-color: #25D366;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                font-weight: bold;
            }}
            .container {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 10px;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 10px;
                text-align: center;
                border-radius: 10px 10px 0 0;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #777;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Application Confirmation</h2>
            </div>
            <p>Dear {user_name},</p>
            <p>Thank you for applying for the <strong>{job_title}</strong> position at <strong>{company_name}</strong>. Your application has been successfully received.</p>
            <p>Our HR team will review your profile and get back to you soon.</p>
            <p>In the meantime, if you have any questions, you can directly contact our HR via WhatsApp by clicking the button below:</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{whatsapp_link}" class="button">Contact HR on WhatsApp</a>
            </div>
            <p>Best Regards,<br>{company_name} HR Team</p>
            <div class="footer">
                <p>This is an automated email. Please do not reply directly to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{smtp_from_name} <{smtp_from_email}>"
    message["To"] = recipient_email

    # Plain-text version for email clients that don't support HTML
    text_content = f"""
    Dear {user_name},

    Thank you for applying for the {job_title} position at {company_name}. Your application has been successfully received.

    Our HR team will review your profile and get back to you soon.

    In the meantime, if you have any questions, you can directly contact our HR via WhatsApp:
    {whatsapp_link}

    Best Regards,
    {company_name} HR Team
    """

    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")

    message.attach(part1)
    message.attach(part2)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
