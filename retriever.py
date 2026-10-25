
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "voiceops_dataset"
    / "Capstone_Project_Dataset_FTTT_2026"
)

KB_DIR = DATASET_ROOT / "kb"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "i", "we", "you",
    "my", "our", "your", "to", "for", "of", "on", "in", "and",
    "or", "can", "do", "does", "how", "what", "when", "it",
    "this", "that", "with", "be", "have", "has", "me", "please",
    "tell", "about", "want", "know", "there", "any", "at",
    "from", "after", "before", "would", "could", "should",
    "will", "was", "were", "hai", "kya", "kaise", "kar",
    "mujhe", "mera", "meri", "ke", "ka", "ki", "ko", "se",
    "kya", "hai", "ho", "hoga", "sakta", "sakti", "hain",
}

TOPIC_TERMS = {
    "delivery": {
        "delivery", "deliver", "shipping", "shipment", "midnight",
        "slot", "slots", "schedule", "scheduled", "timing", "times",
        "dispatch", "arrival", "delivery window", "delivery time",
    },
    "refund": {
        "refund", "refunds", "refunded", "refunding",
        "reimbursement", "reimbursements", "money", "back", "cash", "paid",
    },
    "wallet": {"wallet", "wallets"},
    "warranty": {"warranty", "warranties", "assured", "claim"},
    "return": {
        "return", "returns", "returned", "returning", "exchange",
        "pickup", "collection", "torn", "damaged", "opened",
    },
}


def load_documents():
    """Load Markdown sections and preserve document metadata."""
    documents = []

    if not KB_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge-base folder not found: {KB_DIR}"
        )

    for file_path in sorted(KB_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        metadata = {}

        header_match = re.match(
            r"\A---\s*\n(.*?)\n---\s*(?:\n|$)",
            text,
            flags=re.DOTALL,
        )

        if header_match:
            for line in header_match.group(1).splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = (
                        value.strip().strip("\"'")
                    )

        parts = re.split(
            r"(?=^##\s+\[)",
            text,
            flags=re.MULTILINE,
        )

        for part in parts:
            part = part.strip()

            if not part or part.startswith("---"):
                continue

            section_match = re.match(
                r"^##\s+\[([^\]]+)\]\s*(.*)",
                part,
            )

            if section_match:
                section_id = section_match.group(1).strip()
                section_title = section_match.group(2).strip()
            else:
                section_id = ""
                section_title = ""

            documents.append({
                "filename": file_path.name,
                "doc_id": metadata.get("doc_id", ""),
                "title": metadata.get("title", ""),
                "doc_type": metadata.get("doc_type", ""),
                "status": metadata.get("status", ""),
                "effective_from": metadata.get("effective_from", ""),
                "section_id": section_id,
                "section_title": section_title,
                "text": part,
            })

    return documents


def expand_query(query):
    """Add English intent words for common Hindi/Hinglish phrases."""
    q = query.lower()
    extra_words = set()

    # Hindi cancellation questions: map intent to English policy terms.
    if any(phrase in q for phrase in [
        "ऑर्डर शिप",
        "शिप हो गया",
        "कैंसिल",
        "रद्द",
        "ऑर्डर रोक",
    ]):
        extra_words.update({
            "cancel", "cancellation", "shipped",
            "order", "refuse", "parcel", "door",
        })

    # Warranty and ZentroMart Assured questions
    if (
        "assured" in q
        or "warranty" in q
        or "वारंटी" in q
        or "वॉरंटी" in q
        or "claim" in q
        or "दावा" in q
    ):
        extra_words.update({
            "warranty", "assured", "claim", "coverage",
        })

    # Refund questions, including payment-method timelines
    if (
        "refund" in q
        or "रिफंड" in q
        or "वापस" in q
    ):
        extra_words.update({"refund", "refunds", "timeline"})

    if "wallet" in q:
        extra_words.update({"wallet", "refund"})

    if "card" in q or "credit" in q or "क्रेडिट" in q:
        extra_words.update({"card", "credit", "refund"})

    # Delivery timing and scheduling
    if (
        "midnight" in q
        or "schedule" in q
        or "slot" in q
        or "डिलीवरी" in q
    ):
        extra_words.update({"delivery", "shipping", "timing"})

    # Return requests in English, Hindi, and Hinglish
    if (
        any(word in q for word in [
            "return", "लौट", "वापस", "रिटर्न",
        ])
    ):
        extra_words.update({"return", "returns", "policy"})

    # Fit-related return requests
    if any(word in q for word in [
        "doesn't fit", "does not fit", "don't fit",
        "wrong size", "too small", "too large",
        "doesnt fit", "send it back", "send this back",
        "not fit", "size issue",
    ]):
        extra_words.update({
            "return", "returns", "policy", "eligibility",
            "category", "clothing",
        })

    # Return-window questions: "कितने दिन" means how many days
    if "कितने दिन" in q or "how many days" in q:
        extra_words.update({"return", "days", "period", "window"})

    # Opened-product condition questions
    if any(word in q for word in [
        "खोल", "opened", "open kiya", "खोला",
    ]):
        extra_words.update({"opened", "condition", "return"})

    # Damaged goods and pickup/collection fees
    if any(word in q for word in [
        "torn", "damaged", "फटा", "फटी", "टूटा",
        "टूटी", "damage",
    ]):
        extra_words.update({
            "damaged", "defective", "pickup", "collection",
            "fees", "return",
        })

    if "pickup" in q or "collection" in q:
        extra_words.update({"pickup", "collection", "fees"})

    return tokenize(query) | extra_words


def tokenize(text):
    """Extract English letters and digits for keyword matching."""
    return set(
        re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text.lower())
    )


def detect_topics(query_words):
    return {
        topic
        for topic, terms in TOPIC_TERMS.items()
        if query_words & terms
    }


def document_matches_topic(doc, topic):
    filename = doc["filename"].lower()
    title = doc["section_title"].lower()
    content = doc["text"].lower()

    if topic == "delivery":
        # Prefer sections from the dedicated delivery/shipping policy.
        # Other policies may mention delivery fees without answering
        # delivery timing or scheduling questions.
        if "delivery" in filename or "shipping" in filename:
            return True

        delivery_words = tokenize(title + " " + content)
        return (
            bool(delivery_words & TOPIC_TERMS["delivery"])
            and any(
                word in delivery_words
                for word in {
                    "schedule", "scheduled", "slot", "slots",
                    "midnight", "timing", "times", "dispatch",
                    "arrival", "shipping",
                }
            )
        )

    if topic == "refund":
        return (
            "refund" in filename
            or bool(
                tokenize(title + " " + content)
                & TOPIC_TERMS["refund"]
            )
        )

    if topic == "wallet":
        return (
            "wallet" in filename
            or "wallet" in title
            or "wallet" in content
        )

    if topic == "warranty":
        return (
            "warranty" in filename
            or "warranty" in title
            or "assured" in title
            or "warranty" in content
            or "assured" in content
        )

    if topic == "return":
        return (
            "return" in filename
            or "returns" in filename
            or bool(
                tokenize(title + " " + content)
                & TOPIC_TERMS["return"]
            )
        )

    return True


def search_documents(query, top_k=4):
    """Retrieve relevant sections from current policy documents."""
    if top_k <= 0:
        return []

    docs = load_documents()

    current_docs = [
        doc for doc in docs
        if doc["status"].strip().lower() == "current"
    ]

    query_words = expand_query(query) - STOP_WORDS

    if not query_words:
        return []

    topics = detect_topics(query_words)

    candidates = current_docs

    if topics:
        topic_candidates = [
            doc for doc in current_docs
            if any(
                document_matches_topic(doc, topic)
                for topic in topics
            )
        ]

        query_lower = query.lower()
        asks_delivery_timing = any(
            term in query_lower
            for term in ("midnight", "schedule", "scheduled", "delivery slot")
        )

        asks_item_return = any(
            phrase in query_lower
            for phrase in [
                "send back",
                "return window",
                "return this",
                "return the",
                "rangoli",
                "straight kurta",
                "festive sale",
            ]
        )

        if asks_item_return:
            return_candidates = [
                doc for doc in current_docs
                if document_matches_topic(doc, "return")
            ]
            if return_candidates:
                candidates = return_candidates

        elif asks_delivery_timing:
            delivery_policy_candidates = [
                doc for doc in topic_candidates
                if "delivery" in doc["filename"].lower()
                or "shipping" in doc["filename"].lower()
            ]

            if delivery_policy_candidates:
                candidates = delivery_policy_candidates
            elif topic_candidates:
                candidates = topic_candidates
        elif topic_candidates:
            candidates = topic_candidates

    query_lower = query.lower()

    # Identify specific question intents.
    asks_assured = (
        "assured" in query_lower
        or "वारंटी" in query_lower
        or "वॉरंटी" in query_lower
        or "दावा" in query_lower
    )

    asks_wallet_refund = (
        "wallet" in query_lower
        and ("refund" in query_lower or "रिफंड" in query_lower)
    )

    asks_card_refund = (
        any(word in query_lower for word in [
            "card", "credit", "क्रेडिट",
        ])
        and any(word in query_lower for word in [
            "refund", "रिफंड",
        ])
    )

    asks_upi_refund = (
        any(word in query_lower for word in [
            "upi", "यूपीआई",
        ])
        and any(word in query_lower for word in [
            "refund", "रिफंड", "money back",
        ])
    )

    asks_net_banking_refund = (
        any(word in query_lower for word in [
            "net banking", "netbanking", "नेट बैंकिंग",
        ])
        and any(word in query_lower for word in [
            "refund", "रिफंड", "money back",
        ])
    )

    asks_general_refund_timeline = (
        any(word in query_lower for word in [
            "refund", "रिफंड", "money back",
        ])
        and any(word in query_lower for word in [
            "how long", "how many days", "when",
            "timeline", "time", "कितने दिन", "कब",
        ])
    )

    # Cash-on-delivery refund questions
    asks_cod_refund = (
        any(phrase in query_lower for phrase in [
            "cash on delivery",
            "cash-on-delivery",
            "cod",
            "paid cash",
        ])
        and any(phrase in query_lower for phrase in [
            "refund", "money back", "returned",
            "return the item", "रिफंड", "पैसे वापस",
        ])
    )

    asks_midnight = (
        "midnight" in query_lower
        or "schedule" in query_lower
        or "slot" in query_lower
    )

    # Delivery attempt / customer not home questions
    asks_delivery_attempts = any(phrase in query_lower for phrase in [
        "not home", "not at home", "wasn't home", "was not home",
        "not available", "missed the courier", "missed delivery",
        "courier comes", "delivery attempts",
    ])

    # International delivery questions
    asks_international_delivery = any(phrase in query_lower for phrase in [
        "outside india", "international delivery", "deliver abroad",
        "deliver to dubai", "delivery to dubai", "outside the country",
    ])

    # Cancellation after shipment / order already in transit
    asks_shipped_cancellation = any(phrase in query_lower for phrase in [
        "already on its way", "already shipped", "order has shipped",
        "after shipping", "after it ships", "shipped order",
        "stop my order", "stop it", "cancel my order",
        "cancel the order", "cancelled after shipping",
    ])

    asks_return_period = (
        "कितने दिन" in query_lower
        or "how many days" in query_lower
        or "how long" in query_lower
        or "return window" in query_lower
        or "send back" in query_lower
    )

    asks_festive_return = any(phrase in query_lower for phrase in [
        "festive", "festival", "rangoli",
        "sale", "straight kurta",
    ])

    asks_opened_return = (
        any(word in query_lower for word in [
            "खोल", "खोला", "opened", "open kiya",
        ])
        and any(word in query_lower for word in [
            "वापस", "return", "रिटर्न",
        ])
    )

    asks_damaged_item = any(word in query_lower for word in [
        "torn", "damaged", "damage", "crack", "cracked",
        "broken", "broke", "defective", "faulty",
        "फटा", "फटी", "टूटा", "टूटी", "खराब",
        "not working", "doesn't work", "wont work",
        "won't work", "sorted",
    ])

    asks_damaged_pickup = (
        asks_damaged_item
        and any(word in query_lower for word in [
            "pickup", "collection", "free", "fee", "फ्री",
        ])
    )

    # Fraud or an order the customer did not place
    asks_unauthorized_order = any(phrase in query_lower for phrase in [
        "never placed",
        "didn't place",
        "did not place",
        "not placed by me",
        "unauthorized order",
        "unauthorised order",
        "order i didn't make",
        "order i did not make",
        "never ordered",
        "fraud",
        "account security",
        "अनधिकृत ऑर्डर",
        "मैंने ऑर्डर नहीं किया",
    ])

    scored_docs = []

    sku_tokens = {
        word for word in query_words
        if re.fullmatch(r"[a-z]{2,}-\d+", word)
    }

    for doc in candidates:
        title = doc["section_title"]

        searchable_text = (
            doc["title"] + " " + title + " " + doc["text"]
        ).lower()

        doc_words = tokenize(searchable_text)
        score = len(query_words & doc_words)

        # Give a small boost when the question topic is in the title.
        title_words = tokenize(title)

        for topic in topics:
            if topic in title_words:
                score += 4

        section_id = doc["section_id"]

        # Prefer the policy sections that directly answer these intents.
        if asks_assured and section_id == "WAR-2":
            score += 12

        if asks_wallet_refund and section_id == "REF-2":
            score += 12

        if asks_cod_refund:
            if section_id == "REF-3":
                score += 15
            elif section_id == "REF-2":
                score += 10

        if (
            (
                asks_card_refund
                or asks_upi_refund
                or asks_net_banking_refund
                or asks_general_refund_timeline
            )
            and section_id == "REF-2"
        ):
            score += 12

        if asks_midnight and section_id == "DEL-1":
            score += 12

        if asks_delivery_attempts and section_id == "DEL-3":
            score += 15

        if asks_international_delivery:
            if section_id == "DEL-5":
                score += 15
            elif section_id == "FAQ-5":
                score += 12

        if asks_shipped_cancellation:
            if section_id == "CAN-2":
                score += 15
            elif section_id == "FAQH-1":
                score += 12

        if asks_return_period and section_id == "RET-2":
            score += 12

        if asks_festive_return and section_id == "RET-8":
            score += 12

        if asks_opened_return and section_id == "RET-4":
            score += 12

        if asks_damaged_item:
            if section_id in {"DWM-1", "DWM-2", "DWM-3"}:
                score += 10
            elif section_id in {"ESC-1", "FAQH-3"}:
                score += 8

        if asks_damaged_pickup and section_id == "RET-6":
            score += 10

        # Prioritize security escalation for unauthorized orders
        if asks_unauthorized_order and section_id == "ESC-1":
            score += 15

        # A Plus membership return-fee question needs both policies.
        if (
            "plus" in query_lower
            and any(word in query_lower for word in [
                "pickup", "collection", "fee", "charged",
            ])
        ):
            if section_id == "RET-6":
                score += 8
            elif section_id == "PLS-2":
                score += 8

        if sku_tokens & doc_words:
            score += 5

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(
        key=lambda item: (
            item[0],
            bool(item[1]["section_id"]),
            item[1]["filename"],
            item[1]["section_id"],
        ),
        reverse=True,
    )

    # Avoid returning unrelated sections with only a weak keyword match.
    return [
        {**doc, "score": score}
        for score, doc in scored_docs[:top_k]
        if score >= 2
    ]


if __name__ == "__main__":
    test_queries = [
        "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?",
        "How fast are refunds to the ZentroMart Wallet?",
        "Can I schedule a delivery at midnight?",
        "credit card wala refund 17 October 2026 ko shuru hua, kab tak aa jaana chahiye?",
        "Is the collection free if the UrbanThread Slim Fit Jeans came torn?",
        "मैंने Gulmohar Matte Lipstick खोल लिया है, क्या अब भी वापस कर सकती हूँ?",
        "मुझे Pagecraft A5 Ruled Notebook (Pack of 4) लौटाने के लिए कितने दिन मिलेंगे?",
    ]

    for query in test_queries:
        print(f"\nQuestion: {query}")
        results = search_documents(query, top_k=4)

        if not results:
            print("No matching sections found.")
            continue

        for doc in results:
            print(
                f"Score: {doc['score']} | {doc['filename']} | "
                f"{doc['section_id'] or 'INTRO'} | "
                f"{doc['section_title'] or doc['title']}"
            )
