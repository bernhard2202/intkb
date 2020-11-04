import numpy as np
import torch
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, X, y):
        self.X = X
        self.max = 0
        self.y = y

    def _mrr(self, y, scores):
        scores = np.squeeze(scores[0:10]).numpy()
        if len(np.shape([scores])) == 1:
            return 0
        argsorted_scores = np.argsort(-scores)
        for i, pos in enumerate(argsorted_scores):
            if y[pos] == 1:
                return 1/(pos+1)
        return 0

    def _hitsat(self, y, scores, top):
        scores = np.squeeze(scores[0:10]).numpy()
        if len(np.shape([scores])) == 1:
            if len(y) == 1 and y[0] == 0:
                return 0
            if len(y) == 1 and y[0] == 1:
                return 1
        argsorted_scores = np.argsort(-scores)
        for i, pos in enumerate(argsorted_scores):
            if i == top:
                break
            if y[pos] == 1:
                return 1
        return 0

    def _thres(self, y, scores, th):
        scores = np.squeeze(scores).numpy()
        if len(np.shape([scores])) == 1:
            return 0, 0
        scores = scores/sum(scores)
        argsorted_scores = np.argsort(scores)
        if scores[argsorted_scores[-1]] >= th:
            if y[argsorted_scores[-1]] == 1:
                return 1, 1
            else:
                return 1, 0
        return 0, 0


class EvaluatorKnown(Evaluator):
    def __init__(self, X, y, frequencies):
        super().__init__(X, y)
        self.frequencies = frequencies
        assert len(X) == len(frequencies), (len(X), len(frequencies))

    def evaluate_(self, model, name, range_min, range_top):
        baseline_performance = 0
        baseline_performance5 = 0
        N = 0
        hits1 = 0
        result = []
        hits3 = 0
        hits5 = 0
        mrr = 0
        mrrBL = 0
        print('Size X:', len(self.X))
        with torch.no_grad():
            for i, x in tqdm(enumerate(self.X)):
                if self.frequencies[i] < range_min:
                    continue
                if self.frequencies[i] > range_top:
                    continue
                inputs = []

                for j, candidate in enumerate(x):
                    inputs.append(candidate)

                baseline_performance += int(self.y[i][0])
                bl5 = 0
                for j in range(min(5, len(self.y[i]))):
                    if self.y[i][j] == 1:
                        bl5 = 1
                baseline_performance5 += bl5

                N += 1
                scores = model.predict(torch.stack(inputs))
                result.append(np.argmax(scores[0:10]))
                hits1 += self._hitsat(self.y[i], scores, 1)
                hits3 += self._hitsat(self.y[i], scores, 3)
                hits5 += self._hitsat(self.y[i], scores, 5)
                mrr += self._mrr(self.y[i], scores)
                pos = -1
                for k,y in enumerate(self.y[i]):
                    if y == 1:
                        pos = k+1
                if pos > 0:
                    mrrBL += 1/pos
            if N == 0:
                print('N == 0')
                return '', result
            log = '\n{}:\tN: {}\tH@1: {:.3f}\tH@3: {:.3f}\tH@5: {:.3f}' \
                  '\tMRR: {:.3f}\tBL@1: {:.3f}\tBL@5: {:.3f}\tBL_MRR: {:.3f}'.format(
                      name, N, hits1/N, hits3/N, hits5/N, mrr/N, baseline_performance/N,
                                                                               baseline_performance5/N,mrrBL/N)
        return log, result

    def evaluate(self, model):
        log = "\nEVALUATION RESULTS ============================================"
        logt, full_res = self.evaluate_(model, 'full', range_min=-1, range_top=99999999)
        log = log + logt
        logt, _ = self.evaluate_(model, 'one-shot', range_min=1, range_top=1)
        log = log + logt
        logt, _ = self.evaluate_(model, 'view-shot', range_min=2, range_top=10)
        log = log + logt
        logt, _ = self.evaluate_(model, 'frequent', range_min=100, range_top=99999999)
        log = log + logt

        return log, full_res


class EvaluatorZs(Evaluator):
    def __init__(self, X, y, is_rare, seen_head, seen_tail):
        super().__init__(X,y)
        self.seen_head = seen_head
        self.seen_tail = seen_tail
        self.is_rare = is_rare
        assert len(self.X) == len(y) == len(is_rare)

    def evaluate_(self, model, name):
        baseline_performance = 0
        baseline_performance5 = 0
        result = []
        N = 0
        hits1 = 0
        hits3 = 0
        hits5 = 0
        mrr = 0
        mrrBL = 0
        with torch.no_grad():
            for i, x in tqdm(enumerate(self.X)):
                if name == 'rare' and not self.is_rare[i]:
                    continue
                if name == 'unseen_head' and self.seen_head[i]:
                    continue
                if name == 'unseen_tail' and self.seen_tail[i]:
                    continue
                if name == 'unseen_heta' and (self.seen_tail[i] or self.seen_head[i]):
                    continue
                inputs = []

                for j, candidate in enumerate(x):
                    inputs.append(candidate)
                baseline_performance += int(self.y[i][0])
                bl5 = 0
                for j in range(min(5, len(self.y[i]))):
                    if self.y[i][j] == 1:
                        bl5 = 1
                baseline_performance5 += bl5

                N += 1

                scores = model.predict(torch.stack(inputs))
                result.append(np.argmax(scores[0:10]))

                hits1 += self._hitsat(self.y[i], scores, 1)
                hits3 += self._hitsat(self.y[i], scores, 3)
                hits5 += self._hitsat(self.y[i], scores, 5)
                mrr += self._mrr(self.y[i], scores)
                pos = -1
                for k,y in enumerate(self.y[i]):
                    if y == 1:
                        pos = k+1
                if pos > 0:
                    mrrBL += 1/pos
            if N == 0:
                print('N == 0')
                return '', result
            log = '\n{}:\tN: {}\tH@1: {:.3f}\tH@3: {:.3f}\tH@5: {:.3f}\tMRR: {:.3f}\tBL@1: {:.3f}\tBL@5: {:.3f}\tBL_MRR: {:.3f}'.format(name, N,
                                                                                                            hits1 / N,
                                                                                                            hits3 / N,
                                                                                                            hits5 / N,
                                                                                                            mrr / N,
                                                                                                            baseline_performance / N,
                                                                                                            baseline_performance5 / N,
                                                                                                            mrrBL / N)
            return log, result

    def evaluate(self, model):
        log = "\nEVALUATION RESULTS ============================================"
        logt, full_result = self.evaluate_(model, 'full')
        log = log + logt
        logt, _ = self.evaluate_(model, 'rare')
        log = log + logt
        logt, _ = self.evaluate_(model, 'unseen_head')
        log = log + logt
        logt, _ = self.evaluate_(model, 'unseen_tail')
        log = log + logt
        logt, _ = self.evaluate_(model, 'unseen_heta')
        log = log + logt
        return log, full_result

