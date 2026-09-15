from Bio import Entrez
from Bio import Medline
import requests

# Required by NCBI
Entrez.email = "your@email.com"


def search_pubmed(query, max_results=5, *args, **kwargs):
    print(f"SEARCHING PUBMED...: {query}")
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
    )

    ids = Entrez.read(handle)["IdList"]

    if not ids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(ids),
        rettype="medline",
        retmode="text",
    )

    records = Medline.parse(handle)

    print(f"Retrieved records : {records}")
    return [
        {
            "pmid": r.get("PMID"),
            "title": r.get("TI"),
            "authors": r.get("AU"),
            "journal": r.get("JT"),
            "year": r.get("DP"),
        }
        for r in records
    ]


def classify_sequence(sequence, *args, **kwargs):

    return {
        "prediction": "DNA",
        "confidence": 0.92,
        "sequence": sequence,
    }


def gene_lookup(gene_name, *args, **kwargs):

    url = (
        f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/"
        f"{gene_name}?expand=1"
    )

    r = requests.get(
        url,
        headers={"Content-Type": "application/json"},
    )

    if not r.ok:
        return {"error": "Gene not found"}

    data = r.json()

    return {
        "id": data["id"],
        "symbol": gene_name,
        "chromosome": data["seq_region_name"],
        "start": data["start"],
        "end": data["end"],
        "strand": data["strand"],
        "description": data.get("description"),
    }


def protein_lookup(protein_name, *args, **kwargs):

    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query={protein_name}&format=json&size=1"
    )

    r = requests.get(url)

    if not r.ok:
        return {"error": "Protein not found"}

    results = r.json()["results"]

    if not results:
        return {"error": "Protein not found"}

    p = results[0]

    return {
        "accession": p["primaryAccession"],
        "protein": p["proteinDescription"]["recommendedName"][
            "fullName"
        ]["value"],
        "organism": p["organism"]["scientificName"],
        "function": p.get("comments", []),
    }


def final_answer():

    return {
        "answer": "This is the final answer from the assistant.",
    }
