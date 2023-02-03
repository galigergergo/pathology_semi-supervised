Dataset Structure:
    - root
        - train
            - class 1
                - img 1
                - img 2
                ...
            - class 2
                - img 1
                - img 2
                ...
            ...
        - val
            - class 1
                - img 1
                - img 2
                ...
            - class 2
                - img 1
                - img 2
                ...
            ...
        - semi_supervised
            - unlabeled
                - img 1
                - img 2
                ...

Train BagNet models in a semi-supervised manner:
	- train the model on the labeled data by running:  main.py [path_to_orig_dataset]
	- repeat:
		- evaluate the model on unlabeled data and extend the original dataset based on
		  predictions by running:  semi_supervised.py [path_to_orig_dataset]
		- retrain the model on the extended dataset by running:  main.py [path_to_ext_dataset]
