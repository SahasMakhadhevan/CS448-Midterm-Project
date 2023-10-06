import bayesian_classifier as bc
import logistic_regression as lr
import support_vector_machines as svm


def main():
    eval_mode = input("Evaluate model? (y/n): ")
    if eval_mode == 'y':
        # bc_accuracy = bc.bc(evaluate=True)
        # lr_accuracy = lr.lr(evaluate=True)
        svm_accuracy = svm.predict_tag(evaluate=True)

        # print accuracy of all three models
        # print("Accuracy of dev set using Bayesian Classifier: ", bc_accuracy)
        # print("Accuracy of dev set using Logistic Regression: ", lr_accuracy)
        print("Accuracy of dev set using SVM: ", svm_accuracy)
    else:
        svm.predict_tag()


if __name__ == "__main__":
    main()
