"""One-time OAuth setup for Gmail mailboxes.

Usage:
    python3 -m src.email.setup --name "work"

Prerequisites:
    1. Create a Google Cloud project: https://console.cloud.google.com
    2. Enable the Gmail API: APIs & Services > Library > Gmail API > Enable
    3. Create OAuth 2.0 credentials:
       - APIs & Services > Credentials > Create Credentials > OAuth client ID
       - Application type: Desktop app
       - Download the JSON and save as: email_client_secret.json (project root)
    4. Run this script with a mailbox name

What it does:
    - Opens a browser window for Google OAuth consent
    - Saves the token to data/email_tokens/{name}.json
    - Creates a row in the mailboxes table in data/emails.db

The client_secret.json is shared across all mailboxes — each mailbox gets
its own token (one per Google account / email address).
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Set up a Gmail mailbox for Diplo")
    parser.add_argument("--name", required=True, help="Mailbox name (e.g. 'work', 'personal')")
    parser.add_argument("--client-secret", default="email_client_secret.json",
                        help="Path to Google OAuth client_secret.json (default: email_client_secret.json)")
    args = parser.parse_args()

    client_secret_path = Path(args.client_secret)
    if not client_secret_path.exists():
        print(f"Error: Client secret not found at {client_secret_path}")
        print()
        print("To set up Gmail integration:")
        print("1. Go to https://console.cloud.google.com")
        print("2. Create a project (or use existing)")
        print("3. Enable Gmail API: APIs & Services > Library > Gmail API")
        print("4. Create OAuth credentials: APIs & Services > Credentials > Create > OAuth client ID > Desktop app")
        print("5. Download the JSON and save as: email_client_secret.json (project root)")
        print("6. Re-run this script")
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Error: google-auth-oauthlib not installed. Run: pip install google-auth-oauthlib")
        sys.exit(1)

    token_dir = Path("data/email_tokens")
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / f"{args.name}.json"

    if token_path.exists():
        print(f"Token already exists at {token_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    scopes = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    ]

    print(f"Opening browser for OAuth consent (mailbox: {args.name})...")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes)
    creds = flow.run_local_server(port=0)

    # Save token
    token_path.write_text(creds.to_json())
    print(f"Token saved to {token_path}")

    # Get the authenticated email address
    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        email_address = profile.get("emailAddress", "")
        print(f"Authenticated as: {email_address}")
    except Exception:
        email_address = ""
        print("Warning: could not fetch email address from Gmail API")

    # Register in the mailboxes table
    try:
        from src.email.cache import EmailCache
        cache = EmailCache()
        cache.add_mailbox(
            name=args.name,
            provider="gmail",
            email_address=email_address,
            token_path=str(token_path),
        )
        cache.close()
        print(f"Mailbox '{args.name}' registered in database")
    except Exception as e:
        print(f"Warning: failed to register mailbox in database: {e}")
        print(f"You may need to add it manually.")

    print()
    print("Setup complete! Add this to your .env if not already there:")
    if email_address:
        print(f"  USER_EMAIL_ADDRESSES={email_address}")
    print()
    print("Restart the bot to start monitoring this mailbox.")


if __name__ == "__main__":
    main()
