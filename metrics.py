import numpy as np

__all__ = ['Metrics']

class Metrics():
    def findMetrics (self,stats):
        epslon = 1.0e-12
        precision = stats['tp'] / (stats['tp'] + stats['fp'] + epslon )
        sensitivity = stats['tp'] / (stats['tp'] + stats['fn'] + epslon )  # recall
        specificity = stats['tn'] / (stats['tn'] + stats['fp'] + epslon )
        accuracy = (stats['tp'] + stats['tn']) / (stats['tp'] + stats['tn'] + stats['fp'] + stats['fn'] + epslon )
        f1 = 2.0 / (1.0/(sensitivity+epslon) + 1.0/(precision + epslon))
        return precision, sensitivity, specificity, accuracy, f1
    
    def computeMetrics (self, labelNames, labelName2labelIndex, testDocs , test_labels, predicted, multiLabel):
        ftCountsByLabel = {}
        for lab in labelNames:
            ftCountsByLabel[lab] = { 'tp' : 0, 'fp' : 0, 'tn' : 0, 'fn' : 0}

        predResults = []
        for j, sentence in enumerate(testDocs):
            predResult = {}
            actualIndices = np.where(test_labels[j] > 0)[0]
            actualLabels = labelNames[actualIndices]
            if (multiLabel):
                predictedIndices = np.where(predicted[j] > 0.5)[0] # prob > 0.5
            else:
                predictedIndices = np.array([np.argmax(predicted[j])])

            predictedProbabilities = predicted[j][predictedIndices]
            predictedLabels = labelNames[predictedIndices]

            for lab in labelNames:
                if ( (lab in actualLabels) and (lab in predictedLabels) ):
                    ftCountsByLabel[lab]['tp'] = ftCountsByLabel[lab]['tp'] + 1     # TP
                if ( (lab in actualLabels) and (lab not in predictedLabels) ):
                    ftCountsByLabel[lab]['fn'] = ftCountsByLabel[lab]['fn'] + 1     # FN
                if ( (lab not in actualLabels) and (lab in predictedLabels) ):
                    ftCountsByLabel[lab]['fp'] = ftCountsByLabel[lab]['fp'] + 1     # FP
                if ( (lab not in actualLabels) and (lab not in predictedLabels) ):
                    ftCountsByLabel[lab]['tn'] = ftCountsByLabel[lab]['tn'] + 1     # TN

            probsForActualLabels = np.zeros_like(actualLabels)
            for i, lab in enumerate(actualLabels):
                probsForActualLabels[i] = predicted[j][labelName2labelIndex[lab]]

            predResult['sampleIndex'] = j
#            predResult['sentence'] = sentence
            predResult['actualLabels'] = actualLabels.tolist()
            predResult['predictedLabels'] = predictedLabels.tolist()
            predResult['predictedProbabilitiesForActualLabels'] = probsForActualLabels.tolist()
            predResult['predictedProbabilitiesPredictedLabels'] = predictedProbabilities.tolist()
#            predResult['allPredictedProbabilities'] = predicted[j].tolist()
            predResults.append(predResult)

        totalFtCounts = { 'tp' : 0, 'fp' : 0, 'tn' : 0, 'fn' : 0}
        for item in ['tp', 'fp', 'tn', 'fn']:
            for lab in labelNames:
                totalFtCounts[item] = totalFtCounts[item] + ftCountsByLabel[lab][item]

        totalFtCounts['precision'], totalFtCounts['sensitivity'], totalFtCounts['specificity'], totalFtCounts['accuracy'], totalFtCounts['f1'] = self.findMetrics (totalFtCounts)
        for lab in labelNames:
            ftCountsByLabel[lab]['precision'], ftCountsByLabel[lab]['sensitivity'], ftCountsByLabel[lab]['specificity'], ftCountsByLabel[lab]['accuracy'], ftCountsByLabel[lab]['f1'] = self.findMetrics (ftCountsByLabel[lab])

        metrics = {}
        metrics['results'] = predResults
        metrics['ftCountsByLabel'] = ftCountsByLabel
        metrics['totalFtCounts'] = totalFtCounts

        return metrics

