#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
from nltk.translate.meteor_score import meteor_score
from typing import Any

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
#METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        pass

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        class _DummyWordNet:
            def synsets(self, *args: Any, **kwargs: Any):
                return []

        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("dog")
            wordnet = wn
        except Exception:
            wordnet = _DummyWordNet()

        for i in imgIds:
            assert(len(res[i]) == 1)
            refs = gts[i]
            hyp = res[i][0]
            if isinstance(refs, str):
                refs = [refs]
            tok_refs = [r.split() if isinstance(r, str) else list(r) for r in refs]
            tok_hyp = hyp.split() if isinstance(hyp, str) else list(hyp)
            try:
                score = round(meteor_score(tok_refs, tok_hyp, wordnet=wordnet), 4)
            except TypeError:
                score = round(meteor_score(tok_refs, tok_hyp), 4)
            scores.append(score)
        #print('{}\n'.format(eval_line))
        #self.meteor_p.stdin.write('{}\n'.format(eval_line))
        #print(self.meteor_p.stdout.readline().strip())

        #for i in range(0,len(imgIds)):
        #    scores.append(float(self.meteor_p.stdout.readline().strip()))
        #score = float(self.meteor_p.stdout.readline().strip())
        #self.lock.release()

        return sum(scores)/len(scores), scores

    def method(self):
        return "METEOR"
