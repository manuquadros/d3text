"""Script to compute useful statistics about the dataset"""

import math
import textwrap
from collections import Counter
from functools import reduce
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
from apiadapters.ncbi.parser import is_scanned
from brenda_references.config import config
from plotnine import (
    aes,
    after_stat,
    coord_cartesian,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_histogram,
    geom_point,
    geom_violin,
    ggplot,
    labs,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_minimal,
    theme_tufte,
)
from tinydb import TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tinydb.table import Document, Table


def hbar() -> None:
    print("-" * 70)


type ReferenceCount = dict[int, set[int]]


def plot_counts(counters: dict[str, Counter]) -> None:
    _labels = []
    _counts = []
    _kind = []
    lim = 20
    for name, counter in counters.items():
        labels, counts = zip(*counter.items())
        _labels.extend(labels)
        _counts.extend(counts)
        _kind.extend([name] * len(labels))

        count_df = pd.DataFrame(data={"id": labels, "frequency": counts})
        count_df.to_csv(f"{name}.csv")
        plot = (
            ggplot(count_df, aes(x="frequency", y=after_stat("density")))
            + geom_histogram(binwidth=1)
            + scale_x_continuous(breaks=range(1, lim + 1, 1))
            + coord_cartesian(xlim=(0, lim))
            + labs(
                title=textwrap.fill(
                    f"Frequency distribution of references for each {name}"
                    " in the dataset",
                    width=40,
                ),
                x="Number of references",
                y=f"Proportion per reference count",
            )
            + theme_minimal()
            + theme(plot_title=element_text(ha="center", ma="center"))
        )
        plot.save(f"{name}.svg")


def entity_stats(docs: list[Document], db: TinyDB) -> dict[str, ReferenceCount]:
    """Extract reference counts for each entity type from a Document list."""
    refcounts = {}
    for doc in docs:
        entity_ids: list[int] = []

        # Bacteria are stored in a document as dictionary {id: name}
        # Strains and enzymes are stored as lists of ids.
        for enttype in ("bacteria", "strains", "enzymes"):
            entities: dict | list = doc.get(enttype, [])
            if type(entities) == dict:
                entity_ids = [int(key) for key in entities.keys()]
            elif type(entities) == list:
                entity_ids = entities
            else:
                entity_ids = []

            for entid in entity_ids:
                if db.table(enttype).contains(doc_id=entid):
                    refcounts.setdefault(enttype, {}).setdefault(
                        entid, set()
                    ).add(doc.doc_id)

        # Relations are stored as dictionaries {"subject": id, "object": id}
        has_enzyme_rels = (
            doc["relations"].get("HasEnzyme", {}) if "relations" in doc else {}
        )

        for rel in has_enzyme_rels:
            refcounts.setdefault("has_enzyme", {}).setdefault(
                (rel["subject"], rel["object"]), set()
            ).add(doc.doc_id)

    return refcounts


def report_reference_counts(
    refcounts: dict[str, ReferenceCount], db: TinyDB
) -> None:
    """Report reference counts for the categories in `refcounts`."""

    def as_counter(rc: ReferenceCount) -> Counter:
        return Counter({eid: len(docs) for eid, docs in rc.items()})

    bacdocs = set().union(*refcounts.get("bacteria", {}).values())
    straindocs = set().union(*refcounts.get("strains", {}).values())

    print("Number of references mentioning bacteria:", len(bacdocs))
    print(
        f"Number of references resolved at the strain level: "
        f"{len(straindocs)} ({len(straindocs) / len(bacdocs):.2%})"
    )

    strains = db.table("strains")
    enzymes = db.table("enzymes")

    strain_counts = as_counter(refcounts.get("strains", {}))
    enzyme_counts = as_counter(refcounts.get("enzymes", {}))
    bacteria_counts = as_counter(refcounts.get("bacteria", {}))
    has_enzyme_counts = as_counter(refcounts.get("has_enzyme", {}))

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r" & & \multicolumn{2}{c}{Reference Counts} \\")
    print(r"\cmidrule(l){3-4}")
    print(r"Entity & $n$ & Mean & Range \\")
    print(r"\midrule")
    for label, counts in (
        ("Bacteria", bacteria_counts),
        ("Strains", strain_counts),
        ("Enzymes", enzyme_counts),
        ("Strain--enzyme relations", has_enzyme_counts),
    ):
        vals = np.array(list(counts.values()))
        print(
            f"{label} & {len(vals)} & {vals.mean():.2f} "
            f"& [{vals.min()}, {vals.max()}] \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Number of bacterial species, strains, enzymes and "
        r"strain--enzyme relations in the fulltext dataset, along with "
        r"their mean reference counts.}"
    )
    print(r"\end{table}")

    hbar()

    print("Most common strains mentioned:")
    top_strains = strain_counts.most_common(n=27)
    top_strains_data = tuple(
        (strains.get(doc_id=strain_id), count)
        for strain_id, count in top_strains
    )

    top_strains_data = tuple(
        {
            "doi": strain.get("doi"),
            "taxon": strain["taxon"]["name"] if strain["taxon"] else "",
            "designations": ", ".join(strain["designations"]),
            "count": count,
        }
        for strain, count in top_strains_data
    )
    top_strains_data = pd.DataFrame(top_strains_data)
    top_strains_data.to_csv("top_strains.csv", index=False)

    for strain_id, count in strain_counts.most_common(10):
        strain = strains.get(doc_id=strain_id)
        print(strain, count)

    hbar()
    mct = sum(count for _, count in strain_counts.most_common(27))
    total = strain_counts.total()
    print(f"{mct}/{total} = {mct / total:.2f}")

    freqdist = Counter(strain_counts.values())
    print(freqdist)

    print("Most common enzymes mentioned:")
    for enzyme_id, count in enzyme_counts.most_common(10):
        enzyme = enzymes.get(doc_id=enzyme_id)
        print(f"{enzyme['ec_class']}\t{enzyme['recommended_name']}\t{count}")

    has_enzyme_rc = refcounts.get("has_enzyme", {})

    related_enzymes = Counter()
    for (_, enzyme_id), count in has_enzyme_counts.items():
        related_enzymes[enzyme_id] += count
    hapax_enzymes = [enz for enz, val in related_enzymes.items() if val == 1]

    print("Hapax enzymes:", len(hapax_enzymes))
    print(hapax_enzymes[:10])

    plot_counts(
        {
            "strain": strain_counts,
            "enzyme": enzyme_counts,
            "bacteria": bacteria_counts,
            "strain-enzyme relations": has_enzyme_counts,
        }
    )
    hbar()

    print(
        "Number of strain-enzyme relation instances:",
        has_enzyme_counts.total(),
    )
    print("Number of unique strain-enzyme relations:", len(has_enzyme_counts))

    print("Most common enzyme-strain relations:")
    for (strain_id, enzyme_id), freq in has_enzyme_counts.most_common(5):
        strain = strains.get(doc_id=strain_id)
        enzyme = enzymes.get(doc_id=enzyme_id)
        print()
        print(f"{strain}\n{enzyme['ec_class'], enzyme['recommended_name']}")
        print(freq)
        print()
    print("\n")

    related_strains = Counter(rel[0] for rel in has_enzyme_rc.keys())

    plot_counts({"enzyme": related_enzymes})

    top_strains = related_strains.most_common(
        math.ceil(len(related_strains) * 0.01)
    )
    enzyme_ratio = 0.03
    top_enzymes = related_enzymes.most_common(
        math.ceil(len(related_enzymes) * enzyme_ratio)
    )

    print(
        f"The 1% ({len(top_strains)}) most commonly related strains account "
        f"for {sum(c[1] for c in top_strains) / related_strains.total():.2%} "
        "of all relations."
    )

    print(
        f"The {enzyme_ratio:.2%} ({int(len(related_enzymes) * enzyme_ratio)})"
        " most commonly related enzymes account "
        f"for {sum(c[1] for c in top_enzymes) / related_enzymes.total():.2%}"
        " of all relations."
    )


def main() -> None:
    with TinyDB(
        config["documents"], storage=CachingMiddleware(JSONStorage)
    ) as docdb:
        documents = docdb.table("documents")
        enzymes = docdb.table("enzymes")
        strains = docdb.table("strains")

        print("Number of references:", len(documents))

        # references_without_abstract = tuple(
        #     doc for doc in documents if not doc.get("abstract")
        # )
        # print(
        #     "Number of references without an abstract:",
        #     len(references_without_abstract),
        # )

        # pmc_open = documents.search(where("pmc_open") == True)
        # print("Number of open access references:", len(pmc_open))
        fulltext = documents.search(
            where("fulltext").exists() & (where("fulltext") != "")
        )
        scanned = reduce(
            lambda sum, _: sum + 1,
            filter(lambda doc: is_scanned(doc["fulltext"]), fulltext),
            0,
        )
        print("Full text articles: ", len(fulltext))
        print(
            f"Some of which, {scanned}, are only available as scanned images."
        )

        fulltext_entity_stats = entity_stats(fulltext, docdb)
        report_reference_counts(fulltext_entity_stats, docdb)

        # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        # ax.hist(related_strains.values())
        # fig.savefig("plot.png")  # save the figure to file
        # plt.close(fig)  # close the figure window


if __name__ == "__main__":
    main()
