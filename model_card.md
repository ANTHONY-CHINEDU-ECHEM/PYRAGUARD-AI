# PyraGuard AI model and system card

## Intended use

PyraGuard is decision support for trained site operators and fire wardens. It watches camera feeds for flame, smoke and abnormal heat, and when it confirms an incident it presents a response plan in which every line cites published guidance or the site escalation policy.

It is a supplement to, never a replacement for, a fire detection and alarm system designed and maintained to the applicable standard (in the United Kingdom, BS 5839 Part 1), and it does not replace the fire risk assessment or the emergency plan required by law.

## Out of scope

* Use as the only means of fire detection or of giving warning.
* Automatic actuation of suppression, door release or evacuation signals without a human decision.
* Sites where the knowledge base has not been reviewed and approved by a competent person.

## Components

* Learned detector: Ultralytics YOLO (YOLO26 small by default) fine tuned on DFire, two classes (smoke, fire). Weights are not shipped; train with `make train`.
* Classical detector: chromatic rules plus region scoring; no weights.
* Thermal analyser: thresholding of radiometric frames.
* Retrieval: hashing TF IDF vectors and BM25 by default; Sentence Transformers and OpenCLIP optional.
* Generation: extractive by default; Anthropic, OpenAI or a local Ollama model optional.

## Training data

DFire (Gaia, solutions on demand): 21,527 images; 1,164 with fire only, 5,867 with smoke only, 4,658 with both and 9,838 with neither; 14,692 fire boxes and 11,865 smoke boxes in YOLO format. Cite de Venancio, Lisboa and Barbosa, Neural Computing and Applications, 2022 when using it.

## Evaluation in this repository

All figures in the README come from `reports/benchmark.json` and `reports/rag_eval.json`, produced by `make benchmark` and `make ragcheck`.

* The vision figures were measured with the classical detector on procedurally generated scenes. They test pipeline logic with exact ground truth. They are not evidence of performance on real fires.
* The golden questions and incident scenarios were written alongside the knowledge base and were used during development. They are a regression suite, not a held out test.
* No YOLO accuracy is reported, because none was measured for this repository. `make validate` writes it to `reports/yolo_validation.json` after training.

## Known limitations

* The classical detector cannot see smoke in a single still and raises false candidates on warm lamps and hi vis clothing; about one in nine hard negative clips still opened an incident in the benchmark.
* Generated scenes are stylised. Real smoke moves more slowly than the generated plumes, so the churn threshold may need retuning on real footage.
* The handcrafted image descriptor used for reference matching is coarse (about 65 percent scene accuracy on generated frames). Use the OpenCLIP backend where picture matching matters.
* Extractive answers contained the expected fact for 87.5 percent of the golden questions; the remainder retrieved the right document but selected the wrong sentences.
* The knowledge base is an original summary corpus written for this project from United Kingdom guidance and standards. It has not been reviewed by a fire safety professional and reflects one jurisdiction.
* Evacuation routing is only as good as the site survey behind it.

## Safety design

Every plan line carries citations; unsupported language model output is removed; firefighting advice is removed once a fire is growing; procedures for other room types cannot leak into a plan; an open incident never steps down without a person.

## Licences

PyraGuard code: MIT. Ultralytics YOLO: AGPL 3.0 or a commercial licence. DFire: see the dataset repository for its terms.
