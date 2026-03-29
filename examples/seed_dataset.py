"""Seed the 'text2sql-agent' LangSmith dataset with Chinook DB examples."""

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client

client = Client()

DATASET_NAME = "text2sql-agent"

# Examples based on the Chinook music database
EXAMPLES = [
    {
        "inputs": {"question": "How many albums does AC/DC have?"},
        "outputs": {
            "response": "AC/DC has 2 albums.",
            "sql": "SELECT COUNT(*) FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
        },
    },
    {
        "inputs": {"question": "What are the names of all employees?"},
        "outputs": {
            "response": "The employees are Andrew Adams, Nancy Edwards, Jane Peacock, Margaret Park, Steve Johnson, Michael Mitchell, Robert King, and Laura Callahan.",
            "sql": "SELECT FirstName, LastName FROM Employee;",
        },
    },
    {
        "inputs": {"question": "How many tracks are in the 'Rock' genre?"},
        "outputs": {
            "response": "There are 1297 tracks in the Rock genre.",
            "sql": "SELECT COUNT(*) FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
        },
    },
    {
        "inputs": {"question": "What is the total revenue from all invoices?"},
        "outputs": {
            "response": "The total revenue from all invoices is $2328.60.",
            "sql": "SELECT SUM(Total) FROM Invoice;",
        },
    },
    {
        "inputs": {
            "question": "Which country has the most customers?"
        },
        "outputs": {
            "response": "The USA has the most customers with 13.",
            "sql": "SELECT BillingCountry, COUNT(*) as CustomerCount FROM Invoice GROUP BY BillingCountry ORDER BY CustomerCount DESC LIMIT 1;",
        },
    },
    {
        "inputs": {"question": "List the top 5 longest tracks by duration."},
        "outputs": {
            "response": "The top 5 longest tracks are Occupation / Precipice (5286953ms), Through a Looking Glass (5088838ms), Greetings from Earth, Pt. 1 (2960293ms), The Man With Nine Lives (2956998ms), and Take the Celestra (2927802ms).",
            "sql": "SELECT Name, Milliseconds FROM Track ORDER BY Milliseconds DESC LIMIT 5;",
        },
    },
    {
        "inputs": {
            "question": "How many customers are from Brazil?"
        },
        "outputs": {
            "response": "There are 5 customers from Brazil.",
            "sql": "SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil';",
        },
    },
    {
        "inputs": {
            "question": "What are the different genres available in the database?"
        },
        "outputs": {
            "response": "The genres available are Rock, Jazz, Metal, Alternative & Punk, Rock And Roll, Blues, Latin, Reggae, Pop, Soundtrack, Bossa Nova, Easy Listening, Heavy Metal, R&B/Soul, Electronica/Dance, World, Hip Hop/Rap, Science Fiction, TV Shows, Sci Fi & Fantasy, Drama, Comedy, Alternative, Classical, and Opera.",
            "sql": "SELECT Name FROM Genre;",
        },
    },
]


def seed_dataset():
    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Found existing dataset: {DATASET_NAME} (id: {dataset.id})")
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Evaluation dataset for the text2sql agent using the Chinook music database",
        )
        print(f"Created dataset: {DATASET_NAME} (id: {dataset.id})")

    # Add examples
    client.create_examples(
        inputs=[ex["inputs"] for ex in EXAMPLES],
        outputs=[ex["outputs"] for ex in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"✅ Added {len(EXAMPLES)} examples to '{DATASET_NAME}'")


if __name__ == "__main__":
    seed_dataset()
