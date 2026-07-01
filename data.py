"""
Knowledge base for the RAG application.

Each document represents a self-contained knowledge article.
The metadata can later be used for filtering, citations,
or displaying document sources.
"""

documents = [
    {
        "id": 1,
        "title": "Track an Order",
        "category": "Orders",
        "source": "Amazon Help Center",
        "text": (
            "To track an Amazon order, sign in to your Amazon account and "
            "go to Your Orders. Locate the order you want to track and "
            "select Track Package. You'll see the latest shipping status, "
            "estimated delivery date, carrier information, and delivery "
            "updates if available."
        ),
    },
    {
        "id": 2,
        "title": "Return Policy",
        "category": "Returns",
        "source": "Amazon Help Center",
        "text": (
            "Most Amazon items can be returned within 30 days of delivery. "
            "Items should generally be unused and returned with their "
            "original packaging and accessories. Refund eligibility and "
            "return windows may vary depending on the product category or seller."
        ),
    },
    {
        "id": 3,
        "title": "Return an Order",
        "category": "Returns",
        "source": "Amazon Help Center",
        "text": (
            "To return an item, go to Your Orders, select the order, "
            "choose Return or Replace Items, select a reason for the return, "
            "choose a return method, and follow the instructions. "
            "Refunds are typically processed after the returned item is received."
        ),
    },
    {
        "id": 4,
        "title": "Contact Customer Service",
        "category": "Support",
        "source": "Amazon Help Center",
        "text": (
            "If you need help with an Amazon order or account, visit the "
            "Help section on the Amazon website or mobile app. Customer "
            "service is available through live chat, phone, or email "
            "depending on your region."
        ),
    },
    {
        "id": 5,
        "title": "Amazon Prime Benefits",
        "category": "Amazon Prime",
        "source": "Amazon Help Center",
        "text": (
            "Amazon Prime membership includes fast delivery on eligible items, "
            "Prime Video streaming, Prime Music, Prime Reading, "
            "Prime Gaming, exclusive member-only deals, Prime Day offers, "
            "and Amazon Fresh delivery in eligible locations. "
            "Available benefits may vary by country."
        ),
    },
    {
        "id": 6,
        "title": "Delayed Delivery",
        "category": "Shipping",
        "source": "Amazon Help Center",
        "text": (
            "If your package is delayed, check the latest delivery estimate "
            "from Your Orders. Shipping delays may occur due to weather, "
            "carrier issues, or inventory availability. If the package "
            "does not arrive after the expected delivery date, contact "
            "Amazon Customer Service."
        ),
    },
    {
        "id": 7,
        "title": "Cancel an Order",
        "category": "Orders",
        "source": "Amazon Help Center",
        "text": (
            "To cancel an order, go to Your Orders, select the order, and "
            "choose Cancel Items. Orders that have already shipped may "
            "not be cancelled and may instead need to be returned after delivery."
        ),
    },
    {
        "id": 8,
        "title": "Amazon Gift Cards",
        "category": "Gift Cards",
        "source": "Amazon Help Center",
        "text": (
            "Amazon Gift Cards can be purchased from the Gift Cards section. "
            "Choose a design, delivery method, and amount before adding the "
            "gift card to your cart and completing checkout."
        ),
    },
    {
        "id": 9,
        "title": "Manage Payment Methods",
        "category": "Payments",
        "source": "Amazon Help Center",
        "text": (
            "To add, remove, or update a payment method, sign in to your "
            "Amazon account, open Your Account, and select Your Payments. "
            "You can manage credit cards, debit cards, and other supported "
            "payment methods."
        ),
    },
    {
        "id": 10,
        "title": "Sign In to Your Amazon Account",
        "category": "Account",
        "source": "Amazon Help Center",
        "text": (
            "To sign in to Amazon, select Sign In and enter your registered "
            "email address or mobile phone number along with your password. "
            "If you cannot remember your password, select Forgot Password "
            "to reset it."
        ),
    },
]
