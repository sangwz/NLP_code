from nametocountry_normal import *

def predict_rnn_normal():
# if command == 'predict':
    rnn = torch.load('./ModelOfName/rnn-normal.pkl')
    rnn_result_list = predict_test('rnn')
    lstm = torch.load('./ModelOfName/lstm-normal.pkl')
    lstm_result_list = predict_test('lstm')
    gru = torch.load('./ModelOfName/gru-normal.pkl')
    gru_result_list = predict_test('gru')

def train_rnn_normal():
    all_losses1, period1, rnn = train(trainRNN)
    torch.save(rnn, './ModelOfName/rnn-normal.pkl')
    all_losses2, period2, lstm = train(trainLSTM)
    torch.save(lstm, './ModelOfName/lstm-normal.pkl')
    all_losses3, period3, gru = train(trainGRU)
    # torch.save(rnn, './ModelOfName/rnn-normal.pkl')
    torch.save(gru, './ModelOfName/gru-normal.pkl')
    # 创建画布0
    plt.figure(0)
    # 绘制损失对比曲线
    plt.plot(all_losses1, label="RNN")
    plt.plot(all_losses2, color="red", label="LSTM")
    plt.plot(all_losses3, color="orange", label="GRU")
    plt.legend(loc='upper left')


    # 创建画布1
    plt.figure(1)
    # x_data=["RNN", "LSTM", "GRU"]
    x_data=["RNN"]
    # y_data = [period1, period2, period3]
    y_data = [period1]
    # 绘制训练耗时对比柱状图
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.show()

if __name__ == '__main__':
    train_rnn_normal()
    predict_rnn_normal()
