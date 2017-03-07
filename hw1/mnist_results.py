from torch.autograd import Variable
import pandas as pd
import sys
import pickle
import data_provider


def predict_model(model, pred_data_loader):
    model.start_prediction()
    pred_loader = pred_data_loader.loader
    label_predict = []
    for _, content in enumerate(pred_loader):
        data = Variable(content[0])
        output = model.predict_batch(data)
        pred = output[1].numpy().tolist()
        label_predict += [x[0] for x in pred]
    df = pd.DataFrame({"ID": range(len(label_predict)), "label": label_predict})
    return df

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python mnist_results.py [model_file] [output_df] [test_data_pickle_file]')
        sys.exit(0)
    else:
        print(sys.argv)
        model_file = sys.argv[1]
        output_file = sys.argv[2]
        test_data_file = sys.argv[3]
        model = pickle.load(open(model_file, "rb"))
        pred_label = data_provider.DataProvider(file_dir=test_data_file, train=False)
        df = predict_model(model, pred_label)
        df.to_csv(output_file, index=False)