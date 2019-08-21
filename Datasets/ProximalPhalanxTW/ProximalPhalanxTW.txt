This series of 11 classification problems were created as part of
Luke Davis's PhD titled "Predictive Modelling of Bone Ageing". They
are all derived from the same images, extracted from Cao et al.
"Digital hand atlas and web-based bone age assessment: system
design and implementation". They are designed to test the efficacy
of hand and bone outline detection and whether these outlines could
be helpful in bone age prediction. Algorithms to automatically
extract the hand outlines and then the outlines of three bones of
the middle finger (proximal, middle and distal phalanges) were
applied to over 1300 images, and three human evaluators labelled
the output of the image outlining as correct or incorrect. This
generated three classification problems:
DistalPhalanxOutlineCorrect; MiddlePhalanxOutlineCorrect; and
ProximalPhalanxOutlineCorrect. The next stage of the project was to
use the outlines to predict information about the subjects age. The
three problems DistalPhalanxOutlineAgeGroup,
MiddlePhalanxOutlineAgeGroup and ProximalPhalanxOutlineAgeGroup
involve using the outline of one of the phalanges to predict
whether the subject is one of three age groups: 0-6 years old, 7-12
years old and 13-19 years old. Note that these problems are aligned
by subject, and hence can be treated as a multi dimensional TSC
problem. Problem Phalanges contains the concatenation of all three
problems. Bone age estimation is usually performed by an expert
with an algorithm called Tanner-Whitehouse. This involves scoring
each bone into one of seven categories based on the stage of
development. The final three bone image classification problems,
DistalPhalanxTW, MiddlePhalanxTW and ProximalPhalanxTW, involve
predicting the Tanner-Whitehouse score (as labelled by a human
expert) from the outline.
