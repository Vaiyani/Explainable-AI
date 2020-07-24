# Opinion Lab Group 1.5

## Timetable
|     | 1st      | 2nd      | 3rd      | 4th      | 5th      | 6th      | Final    |
|-----|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Presentation | [27\.04\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_04_27.pptx) | [11\.05\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_05_11.pptx) | [25\.05\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_05_25.pptx) | [08\.06\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_06_08.pptx) | [22\.06\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_06_22.pptx) | [06\.07\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_07_06.pptx) | [20\.07\.](https://gitlab.lrz.de/nlp-lab-course-ss2020/opinion-mining/opinion-lab-group-1.5/-/blob/master/presentations/2020_07_20.pdf) |

## Group Members
- @ge39dor (ge39dor@mytum.de)
- @ga58nav (florian.angermeir@tum.de)

## Initial Action Items
- Apply the given classifier method for attribute and entity classification on SemEval 2016 task 5 and the organic dataset.
  - The idea for SemEval is to show that the classifier is actually working for the task of attribute/entity classification, since the dataset is well defined. Additionally, it would be interesting to analyze the reasoning of the classification to see which are coherent and which are incoherent attribute/entity classes.
- If the classifier can be used for the task, we want to investigate the organic dataset. The problem here is that the annotations are not reliable, i.e. some annotators have different annotation standards than others. Please use the techniques from the paper to explain 
  - which annotators decrease model performance (see the "Annotator" column from the annotated organic dataset),
  - which are the coherently or reliably annotated classes and which are the other classes.
- The main research question to be solved here is how the SS3 classification model can be utilized for our denoted use case. Come up with own ideas in that regard! One option would be to generate the illustrations from figure 6 from the SS3 paper by iteratively adding annotations from one annotators after the other to see who decreases and who increases the confidence values. There will be many other options as well!

## Resources
- [The SS3 Classification Model](https://pyss3.readthedocs.io/en/latest/user_guide/ss3-classifier.html#ss3-introduction)
- [Demos](http://tworld.io/ss3/)
- [Paper](https://arxiv.org/pdf/1905.08772.pdf)
- [Annotated Organic Dataset](https://gitlab.lrz.de/social-rom/organic-dataset/annotated-dataset/-/tree/master/annotated_3rd_round%2Fprocessed%2Ftrain_test_validation%20V0.3)
- [SemEval 2016 task 5](http://alt.qcri.org/semeval2016/task5/)
