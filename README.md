# animal_law_classifier

This repository is referred to from the following manuscript: https://link.springer.com/article/10.1007/s10506-022-09313-y. It contains:
 - A case law repository (case_law_repository.csv)
 - A labelling guidance document (animal_protection_law_guidance.docx)
 - A script that gathers judgment links (scrape_label.py)
 - A script that trains and tests TF-IDF SVM, TF-IDF MLP, USE SVM, USE MLP, s-BERT SVM and s-BERT MLP systems, and creates a case law repository using the best-performing TF-IDF SVM system (train_classify2.py)
 - A script that creates judgment embeddings using BigBird and Longformer (transformer_embedding_extraction.py; produced by Guy Aglionby: https://github.com/GuyAglionby)
 - A script that trains and tests SVM and MLP models using judgment embeddings created using BigBird and Longformer (transformer_classification.py; produced by Guy Aglionby: https://github.com/GuyAglionby)

The case law repository holds links to all judgments containing 'animal' made between January 2000 and December 2020 from the Privy Council, House of Lords, Supreme Court and upper England and Wales courts. Judgment links are provided alongside human- and ML-created labels, showing whether judgments are concerned with animal protection law.

The labelling guidance document contains a docx file that guided the classification of court judgments sampled for human labelling into two categories: those which are substantially concerned with animal protection and those which are not substantially concerned with animal protection.
