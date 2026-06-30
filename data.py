"""
Knowledge base for the RAG application.

This module contains the collection of documents that can be retrieved
to answer user questions. Each document includes metadata such as an
identifier, title, and text content.

In a production system, this data would typically come from a database,
vector store, or external document repository.
"""

documents = [
    {
        "id": 1,
        "title": "Track Order",
        "text": "To track your Amazon order, log into your account, go to 'Your Orders,' and click 'Track Package' for real-time updates.",
    },
    {
        "id": 2,
        "title": "Return Policy",
        "text": "Amazon's return policy allows most items to be returned within 30 days of delivery for a full refund, provided they are in new condition with original packaging and accessories.",
    },
    {
        "id": 3,
        "title": "Return an Order",
        "text": "To return an Amazon order, initiate a return through 'Your Orders,' ship the item back, and receive a refund once processed.",
    },
    {
        "id": 4,
        "title": "Customer Support",
        "text": "To contact Amazon customer service, use the Help section on the website or app to chat, call, or email support.",
    },
    {
        "id": 5,
        "title": "Prime Benefits",
        "text": "Amazon Prime members receive free two-day shipping, exclusive deals, and access to Prime Video and Prime Music.",
    },
    {
        "id": 6,
        "title": "Delayed Package",
        "text": "If your Amazon package is delayed, check the estimated delivery date in Your Orders or contact customer service.",
    },
    {
        "id": 7,
        "title": "Cancel Order",
        "text": "To cancel an Amazon order, go to Your Orders, select the order, and click Cancel Items if it hasn't shipped yet.",
    },
    {
        "id": 8,
        "title": "Gift Cards",
        "text": "To purchase an Amazon gift card, visit the Gift Cards section, choose a design and amount, add it to your cart, and complete checkout.",
    },
    {
        "id": 9,
        "title": "Payment Method",
        "text": "To update your Amazon payment method, go to Your Account, select Your Payments, and add or edit your payment details.",
    },
    {
        "id": 10,
        "title": "Login",
        "text": "To sign in to Amazon, click Sign In and enter your registered email or phone number and password.",
    },
]
