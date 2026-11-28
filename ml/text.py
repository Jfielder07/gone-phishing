"""
How an email is turned into the single piece of text the model reads.

Kept in its own small file (no pandas, no matplotlib) so the website can use it
without the training-only packages. Training and prediction both use these
functions, so the model always sees text built the same way.
"""
MAX_CHARS = 20_000  # very long emails are cut down so training stays fast


def build_text(subject, body):
    """Combine subject and body into the single piece of text the model reads."""
    return f"{subject or ''}\n\n{body or ''}".strip()[:MAX_CHARS]


def email_to_text(email):
    """
    Text for an Email object: subject, body, then any links that aren't already
    written in the body (links hidden behind HTML buttons only show up in email.links).
    """
    body = email.get_body()
    hidden_links = [link for link in email.links if link not in body]
    if hidden_links:
        body = body + "\n" + "\n".join(hidden_links)
    return build_text(email.subject, body)
