# animal_law_classifier

This repository holds:
 - A case law repository
 - A labelling guidance document
 - A script that gathers judgment links
 - A script that trains and applies an animal law judgment classifier, to create a case law repository.

The case law repository holds links to all judgments containing 'animal' made between January 2000 and December 2020 from the Privy Council, House of Lords, Supreme Court and upper England and Wales courts. Judgment links are provided alongside human- and ML-created labels, showing whether judgments are concerned with animal protection law.

The labelling guidance document contains a docx file that to guide the classification of court judgments sampled for human labelling into two categories: those which are substantially concerned with animal protection and those which are not substantially concerned with animal protection.

The script that gathers judgment links is named, scrape_label.py

The script that rains and applies an animal law judgment classifier is named, train_classify2.py
