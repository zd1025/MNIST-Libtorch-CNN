#include <iostream>
#include "../include/model.h"
#include <torch/script.h>
using namespace std;
using namespace chrono;

model::model(){
    cout << "------- model类构造函数被调用 -------" << endl;
}


void model::net_test(){
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在网络测试 -------" << endl;
    MNIST_CNN net;
    cout << "看看网络结构==>" << net << endl;
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- 网络测试已完成，合计花费" << (etime - stime) << "秒 -------" << endl;
}

torch::DeviceType model::device_pick(){
    torch::DeviceType device = torch::kCPU; // 默认为CPU训练
    if(torch::cuda::is_available()){
        cout << "CUDA存在，训练将被放至CUDA进行" << endl;
        device = torch::kCUDA;
    }
    else{
        cout << "CUDA不存在，训练将被放至CPU进行" << endl;
        device = torch::kCPU;
    }
    return device;
}

void model::train_model(){
    auto train_stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在进行网络训练操作 -------" << endl;
    // 进行设备选择
    torch::Device device = device_pick();
    // 模型结构体创建 设置训练设备
    model::MNIST_CNN mnist_cnn;
    mnist_cnn->to(device);

    // 加载数据集
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在进行数据集加载操作 -------" << endl;
    const char *pre_path = "";
    string kDataRoot = pre_path;
    kDataRoot.append("D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\data\\MNIST\\raw");
    // 对MNIST数据进行Normalize和Stack（将多个Tensor stack成一个Tensor)
    auto train_dataset = torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTrain)
            // 进行数据预处理 对灰度图的标准化 能够使得图片特征更加明显 固定计算结果 均值是0.1307，标准差是0.3081
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            // 堆压到栈中方便进行处理
            .map(torch::data::transforms::Stack<>());
    // 求得训练集大小
    const size_t train_dataset_size = train_dataset.size().value();
    // 训练集加载器，根据提供的batch_size大小每次取相应大小的图片
    auto train_loader = torch::data::make_data_loader(move(train_dataset), model::batch_size);
    cout << "训练数据集的长度为==>" << train_dataset_size << endl;

    // 计时器
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- 数据集已加载，合计花费" << (etime - stime) << "秒 -------" << endl;

    // 优化函数初始化 初始化学习率设置为0.001
    torch::optim::Adam optimizer(
    mnist_cnn->parameters(), torch::optim::AdamOptions(model::learning_rate));

    // 交叉熵函数初始化 可能会导致模型的有效范围的缩减 反而影响了网络
//    auto criterion = torch::nn::CrossEntropyLoss();
//    criterion->to(device);

    double running_loss = 0; // 运行过程中总的loss
    for (int64_t epoch = 1; epoch <= model::epoch; ++epoch) {
        int64_t batch_index = 0; // 训练批数
        double correct = 0; // 记录正确个数 设置为double 方便计算准确率
        int64_t round_time = 0; // 本轮中第几次输出
        cout << "------- 第" << epoch << "/" << model::epoch << "轮训练开始 -------" << endl;
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad(); // 清除以往梯度 重新进行计算 对每一轮的结果重新优化 提升参数的精度
            auto output = mnist_cnn->forward(data); // 求得网络结果

            // 获取预测结果
            auto pred = output.argmax(1);
            // 如果每批结果等于标签的结果那么将对应的正确数量增加
            correct += pred.eq(targets).sum().template item<int64_t>();

            // 求出当前的loss值
            auto loss = torch::nll_loss(output, targets, /*weight=*/{}, torch::Reduction::Sum);
            loss.to(device);
            // 检测loss是否为一个数 程序调试和测试函数 如果出现了不符合的情况将会直接报出错误
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            /*
             * 负对数似然损失 常用于分类网络中和log_softmax配合使用
             * 在log情况下的概率小于1为负 使用nll_loss得到的loss为正
             * loss越大代表准确率越高
             * */
            running_loss = running_loss + loss.template item<float>();

            // 将loss进行反向传播的操作 计算其梯度 对其进行参数优化
            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0) { // 每100批输出结果
                round_time++;
                std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f\n",
                        epoch,
                        batch_index * batch.data.size(0),
                        train_dataset_size,
                        loss.template item<float>());
                std::printf("\r第 %ld 轮，第 %ld 次训练集上的Loss总和: %.4f\n",
                        epoch,
                        round_time,
                        running_loss);
                std::printf("\rAcc: %.4f % \n",
                            correct / (batch_index * batch.data.size(0))*100);
            }
        }
        cout << "------- 第" << epoch << "轮训练结束 -------" << endl;
    }

    auto train_etime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 训练已结束，合计花费" << (train_etime - train_stime) << "秒 -------" << endl;

    // 保存最终训练的模型
    save_model(mnist_cnn);
}

void model::test_model(){
    auto test_stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在进行网络测试操作 -------" << endl;
    // 进行设备选择
    torch::Device device = device_pick();
    // 模型结构体创建 设置训练设备
    torch::serialize::InputArchive archive = load_model();
    MNIST_CNN mnist_cnn;
    mnist_cnn->load(archive);
    mnist_cnn->eval();
    mnist_cnn->to(device);

    double test_loss = 0;
    int32_t correct = 0;

    // 加载数据集
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在进行数据集加载操作 -------" << endl;

    const char *pre_path = "";
    string kDataRoot = pre_path;
    kDataRoot.append("D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\data\\MNIST\\raw");
    // 对MNIST数据进行Normalize和Stack（将多个Tensor stack成一个Tensor)
    auto test_dataset = torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), model::batch_size);

    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- 数据集已加载，合计花费" << (etime - stime) << "秒 -------" << endl;

    for (const auto& batch : *test_loader) {
        // 获取数据和label
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        // 模型前向操作，得到预测输出
        auto output = mnist_cnn->forward(data);
        // 计算测试时的 loss
        test_loss += torch::nll_loss(
                output,
                targets,
                /*weight=*/{},
                torch::Reduction::Sum)
                .item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= test_dataset_size;
    std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / test_dataset_size) * 100;

    auto test_etime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 测试已结束，合计花费" << (test_etime - test_stime) << "秒 -------" << endl;
}

void model::save_model(MNIST_CNN& net) {
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在进行模型保存操作 -------" << endl;
    const char *pre_path = "D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\save_model\\";
    // 模型存放路径
    string model_save_path = pre_path;
    model_save_path.append("MNIST_CNN.pt");
    torch::save(net, model_save_path);
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- 模型已保存，合计花费" << (etime - stime) << "秒 -------" << endl;
}

torch::serialize::InputArchive model::load_model() {
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- 正在进行模型加载操作 -------" << endl;
    const char *pre_path = "";
    // 模型存放路径
    string model_load_path = pre_path;
    // 模型加载路径
    model_load_path.append("D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\save_model\\MNIST_CNN.pt");
    torch::serialize::InputArchive archive;
    archive.load_from(model_load_path);
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- 模型已加载，合计花费" << (etime - stime) << "秒 -------" << endl;
    return archive;
}