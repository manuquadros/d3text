from brenda_references.docdb import BrendaDocDB
from tqdm import tqdm
import re

if __name__ == "__main__":
    nondigit = re.compile(r"[^\d]")

    count = 0
    with BrendaDocDB() as docdb:
        for doc in tqdm(docdb.references):
            if doc["pubmed_id"]:
                try:
                    int(doc["pubmed_id"])
                except ValueError:
                    pmid = nondigit.sub("", doc["pubmed_id"])
                    print(f"{doc['pubmed_id']} -> {pmid}")

                    if pmid:
                        docdb.update_record(
                            table="documents",
                            fields={"pubmed_id": pmid},
                            doc_id=doc.doc_id,
                        )
                        count += 1

    if count:
        print(f"{count} documents updated.")
