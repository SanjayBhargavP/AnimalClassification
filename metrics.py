from sklearn.metrics import precision_score, recall_score, f1_score

class Metrics:
    @staticmethod
    def compute_metrics(all_labels, all_preds):
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return precision, recall, f1
