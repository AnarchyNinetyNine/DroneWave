#!/usr/bin/env python3
"""
Request UAV-Gesture Dataset Script
----------------------------------
This script generates and sends a professional dataset request email to
Asanka Perera for academic research purposes. It prompts the user for
their credentials and displays the final email content before sending.
"""

import smtplib
from email.mime.text import MIMEText
from getpass import getpass

def create_email(your_email, your_name, institution, purpose):
    """Generate the MIMEText email for requesting the dataset."""
    message = (
        f"Dear Asanka Perera,\n\n"
        f"My name is {your_name} from {institution}. I am currently conducting research on "
        f"UAV gesture-based navigation systems. As part of this work, I would like to request "
        f"access to the UAV-Gesture dataset for academic research purposes.\n\n"
        f"Purpose of the work: {purpose}\n\n"
        f"Thank you very much for your consideration.\n\n"
        f"Best regards,\n{your_name}"
    )
    msg = MIMEText(message)
    msg['Subject'] = 'Request for UAV-Gesture Dataset'
    msg['From'] = your_email
    msg['To'] = 'asanka.perera@mymail.unisa.edu.au'
    return msg, message

def send_email(msg, your_email, password):
    """Send the prepared email using SMTP."""
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(your_email, password)
        server.send_message(msg)
    print("\n✅ Email sent successfully!")

if __name__ == "__main__":
    print("=== UAV-Gesture Dataset Request ===\n")

    # Prompt user input
    your_email = input("Enter your email address: ").strip()
    password = getpass("Enter your email password (input hidden): ")
    your_name = input("Enter your full name: ").strip()
    institution = input("Enter your institution name: ").strip()
    purpose = input("Briefly describe the purpose of your research: ").strip()

    # Create the email
    msg, final_message = create_email(your_email, your_name, institution, purpose)

    # Display final message for confirmation
    print("\n--- Email Preview ---")
    print(final_message)
    confirm = input("\nDo you want to send this email? (yes/no): ").strip().lower()

    if confirm == 'yes':
        send_email(msg, your_email, password)
    else:
        print("Email not sent. You can modify your inputs and try again.")
