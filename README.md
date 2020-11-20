# Tab2Know

[![Maintenance][maintain-badge]][maintain-act] [![PRs Welcome][pr-badge]][pr-act] [![Build Status][travisci-badge]][travisci-builds]

**Building a Knowledge Base from Tables in Scientific Papers**

#### Table of contents

- :anchor: [Introduction](#anchor-introduction)
- :hammer_and_pick: [Setup](#hammer_and_pick-setup)
- :building_construction: [Development](#building_construction-development)
- :package: [Datasets and Evaluation](#package-datasets-and-evaluation) 
- :triangular_flag_on_post: [Contributing](#triangular_flag_on_post-contributing)
- :memo: [Citation](#memo-citation)

## :anchor: Introduction

[Tab2Know][tab2know] is a system that addresses the challenge of automatically interpreting the tables in papers and of disambiguating the entities they contain. The pipeline embodied employs both statistical-based classifiers and logic-based reasoning.


## :hammer_and_pick: Setup

First, clone the repository and its submodules

```sh
git clone --recurse-submodules -j8 repository-url
```

If `--recurse-submodules` is not available for whatever reason or you have already cloned the repository without this flag, the same effect can be achieved as follows:

1. Clone the repository if you haven't already done so:

   ```sh
   git clone repository-url
   ```
2. Run:

   ```sh
   git submodule update --init
   ```

To set up the extraction, annotation, prediction and evaluation pipeline, refer to the steps below. For additional details, please contact the authors.

1. Download the Semantic Scholar Open Research Corpus, and select the JSON for the papers that you wish to process. Then, convert it to RDF using the `tools/s2-context.jsonld` JSON-LD context.
2. Extract tables from these PDFs using the table extraction Tabula wrapper described below. Example extractions are in the `examples/` directory.
3. Write label queries for weak supervision using the annotation interface described below. Optionally, manually label a sample of tables in this interface or use our set of annotations.
4. Train and evaluate the models using  `python3 tab2know/main.py` on the command line. For help, run `python3 tab2know/main.py -h`.

## :building_construction: Development


### Table extraction

The extractor we offered under the `tools/` directory is a fat jar written in Scala that utilises [Tabula][tabula] for locating the tables in PDFs and [PDFFigures2][pdffigure2] for table extraction.

As the jar contains all used dependencies, you can run the extraction directly by using the following command
```sh
java -jar tab2know-extractor.jar \
    --pdf path-to/pdf/source/ \
    --csv destination-for/extracted/csv/ \
    --img destination-for/extracted/images/ \
    --json destination-for/extracted/json/
```

### Annotation interface

To run the annotation interface, start a SPARQL endpoint to the extracted table structure (we recommend [Trident](https://github.com/karmaresearch/trident)). Then, start the annotation interface with a command similar to the following

```sh
env PYTHONPATH=annotation-webapp/ \ 
    FLASK_ENV=development \
    FLASK_APP=annotation-webapp/app.py \
    BASEDIR=extracted-data/ ANNOTATIONDIR=evaluation/table \
    MODELDIR=models/ \
    QUERYDIR=queries/ \
    KB=http://localhost:19999 \
    ANNOTATION_NAMESPACE=sample \
    flask run --port=5555
``` 

Please refer to `annotation-webapp/app.py` for details.


### Entity linking

With regard to entity linking, we employed [Vlog][vlog]. `tools/vlog_rules` includes the rules deployed. 


## :package: Datasets and Evaluation 

### Datasets

We have made the datasets that we used in our research publicly available at [here][zenodo]. This resource contains the following files:

- `venues.txt`: The venues that were use for selecting PDFs from the [Semantic Scholar Open Research Corpus][s2] that were published in the last 5 years.
- `sample-400.tar.gz`: A sample of the tables that we extracted using [Tabula][tabula] from these PDFs.
- `ontology.ttl`: The annotation ontology in Turtle format.
- `all_metadata.jsonl`: Annotations for this sample in the JSON format described below.
- `labelqueries.csv`: The label queries used for weak annotation, created using the annotation interface. This CSV file contains 6 columns: a numeric ID, the label query template name (`template`), the template slots (`slots`), the label type (`label`), the annotation value (`value`), and a toggle for the interface (`enabled`).
- `labelqueries-sparql-templates.zip`: The label query templates. These are SPARQL queries with slots of the form `{{slot}}`. The templates in `labelqueries.csv` refer to these files.
- `rules.txt`: Datalog rules that we used for entity resolution.
- `tab2know-graph.nt.gz`: The final RDF graph that contains all extracted table structures, predicted table and column classes, and resolved entity links.

### Evaluation

Both annotations of the training set and entity evaluations reside in the `evaluation/` directory. 

The table annotations are of the following format:
```json
{
    "@id": "table-id",
    "@context": [
        "http://www.w3.org/ns/csvw",
        {
            "domain": "domain-uri"
        }
    ],
    "url": "table-url",
    "dc:title": "table-caption",
    "dc:source": "the-semantic-scholar-id-of-the-paper",
    "schema:pagination": 7,
    "schema:position": "7",
    "schema:image": "image-uri",
    "dialect": {
        "headerRowCount": 1,
        "tab2know:isTransposed": 0,
        "tab2know:isMatrix": 1
    },
    "rdf:type": "table-type",
    "tableSchema": {
        "columns": [
            {
                "propertyUrl": "type-of-column-0"
            },
            {
                "propertyUrl": "type-of-column-1"
            }
        ]
    },
    "tab2know:isBad": 0,
    "tab2know:isHardForTabula": 0
}
```

The meaning of the columns in the entity annotations are as follows

* `eid`: the identifier of the entity;
* `is_error`: whether the extracted data is a real entity or not;
* `num_mentions`: the number of times this entity appears in the test set;
* `max_num_same_meaning`: the maximum cardinality of the set that contains the same entities of the same semantic meaning;



## :triangular_flag_on_post: Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. Then, follow the PR template detailing your changes.

## :memo: Citation

```kt
@inproceedings{kruit2020tab2know.
    author="Kruit, Benno and He, Hongyu and Urbani, Jacopo",
    title="Tab2Know: Building a Knowledge Base from Tables in Scientific Papers",
    booktitle="The Semantic Web -- ISWC 2020",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="349--365",
    isbn="978-3-030-62419-4"
}
```


[tab2know]: https://github.com/karmaresearch/tab2know
[travisci-badge]: https://travis-ci.com/HongyuHe/Tab2Know.svg?token=tLQAnpmJrz1TBJtLskoQ&branch=develop
[travisci-builds]: https://travis-ci.com/HongyuHe/Tab2Know
[maintain-badge]: https://img.shields.io/badge/Maintained%3F-yes-green.svg
[maintain-act]: https://github.com/HongyuHe/Tab2Know/graphs/commit-activity
[pr-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
[pr-act]: http://makeapullrequest.com
[tabula]: https://github.com/tabulapdf/tabula
[pdffigure2]: https://github.com/allenai/pdffigures2
[vlog]: https://github.com/jrbn/vlog
[zenodo]: https://doi.org/10.5281/zenodo.3983013
[s2]: http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/