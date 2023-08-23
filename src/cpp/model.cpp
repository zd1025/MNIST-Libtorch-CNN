#include <iostream>
#include "../include/model.h"
#include <torch/script.h>
using namespace std;
using namespace chrono;

model::model(){
    cout << "------- model�๹�캯�������� -------" << endl;
}


void model::net_test(){
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ����������� -------" << endl;
    MNIST_CNN net;
    cout << "��������ṹ==>" << net << endl;
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- �����������ɣ��ϼƻ���" << (etime - stime) << "�� -------" << endl;
}

torch::DeviceType model::device_pick(){
    torch::DeviceType device = torch::kCPU; // Ĭ��ΪCPUѵ��
    if(torch::cuda::is_available()){
        cout << "CUDA���ڣ�ѵ����������CUDA����" << endl;
        device = torch::kCUDA;
    }
    else{
        cout << "CUDA�����ڣ�ѵ����������CPU����" << endl;
        device = torch::kCPU;
    }
    return device;
}

void model::train_model(){
    auto train_stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ���ڽ�������ѵ������ -------" << endl;
    // �����豸ѡ��
    torch::Device device = device_pick();
    // ģ�ͽṹ�崴�� ����ѵ���豸
    model::MNIST_CNN mnist_cnn;
    mnist_cnn->to(device);

    // �������ݼ�
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ���ڽ������ݼ����ز��� -------" << endl;
    const char *pre_path = "";
    string kDataRoot = pre_path;
    kDataRoot.append("D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\data\\MNIST\\raw");
    // ��MNIST���ݽ���Normalize��Stack�������Tensor stack��һ��Tensor)
    auto train_dataset = torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTrain)
            // ��������Ԥ���� �ԻҶ�ͼ�ı�׼�� �ܹ�ʹ��ͼƬ������������ �̶������� ��ֵ��0.1307����׼����0.3081
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            // ��ѹ��ջ�з�����д���
            .map(torch::data::transforms::Stack<>());
    // ���ѵ������С
    const size_t train_dataset_size = train_dataset.size().value();
    // ѵ�����������������ṩ��batch_size��Сÿ��ȡ��Ӧ��С��ͼƬ
    auto train_loader = torch::data::make_data_loader(move(train_dataset), model::batch_size);
    cout << "ѵ�����ݼ��ĳ���Ϊ==>" << train_dataset_size << endl;

    // ��ʱ��
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- ���ݼ��Ѽ��أ��ϼƻ���" << (etime - stime) << "�� -------" << endl;

    // �Ż�������ʼ�� ��ʼ��ѧϰ������Ϊ0.001
    torch::optim::Adam optimizer(
    mnist_cnn->parameters(), torch::optim::AdamOptions(model::learning_rate));

    // �����غ�����ʼ�� ���ܻᵼ��ģ�͵���Ч��Χ������ ����Ӱ��������
//    auto criterion = torch::nn::CrossEntropyLoss();
//    criterion->to(device);

    double running_loss = 0; // ���й������ܵ�loss
    for (int64_t epoch = 1; epoch <= model::epoch; ++epoch) {
        int64_t batch_index = 0; // ѵ������
        double correct = 0; // ��¼��ȷ���� ����Ϊdouble �������׼ȷ��
        int64_t round_time = 0; // �����еڼ������
        cout << "------- ��" << epoch << "/" << model::epoch << "��ѵ����ʼ -------" << endl;
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad(); // ��������ݶ� ���½��м��� ��ÿһ�ֵĽ�������Ż� ���������ľ���
            auto output = mnist_cnn->forward(data); // ���������

            // ��ȡԤ����
            auto pred = output.argmax(1);
            // ���ÿ��������ڱ�ǩ�Ľ����ô����Ӧ����ȷ��������
            correct += pred.eq(targets).sum().template item<int64_t>();

            // �����ǰ��lossֵ
            auto loss = torch::nll_loss(output, targets, /*weight=*/{}, torch::Reduction::Sum);
            loss.to(device);
            // ���loss�Ƿ�Ϊһ���� ������ԺͲ��Ժ��� ��������˲����ϵ��������ֱ�ӱ�������
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            /*
             * ��������Ȼ��ʧ �����ڷ��������к�log_softmax���ʹ��
             * ��log����µĸ���С��1Ϊ�� ʹ��nll_loss�õ���lossΪ��
             * lossԽ�����׼ȷ��Խ��
             * */
            running_loss = running_loss + loss.template item<float>();

            // ��loss���з��򴫲��Ĳ��� �������ݶ� ������в����Ż�
            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0) { // ÿ100��������
                round_time++;
                std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f\n",
                        epoch,
                        batch_index * batch.data.size(0),
                        train_dataset_size,
                        loss.template item<float>());
                std::printf("\r�� %ld �֣��� %ld ��ѵ�����ϵ�Loss�ܺ�: %.4f\n",
                        epoch,
                        round_time,
                        running_loss);
                std::printf("\rAcc: %.4f % \n",
                            correct / (batch_index * batch.data.size(0))*100);
            }
        }
        cout << "------- ��" << epoch << "��ѵ������ -------" << endl;
    }

    auto train_etime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ѵ���ѽ������ϼƻ���" << (train_etime - train_stime) << "�� -------" << endl;

    // ��������ѵ����ģ��
    save_model(mnist_cnn);
}

void model::test_model(){
    auto test_stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ���ڽ���������Բ��� -------" << endl;
    // �����豸ѡ��
    torch::Device device = device_pick();
    // ģ�ͽṹ�崴�� ����ѵ���豸
    torch::serialize::InputArchive archive = load_model();
    MNIST_CNN mnist_cnn;
    mnist_cnn->load(archive);
    mnist_cnn->eval();
    mnist_cnn->to(device);

    double test_loss = 0;
    int32_t correct = 0;

    // �������ݼ�
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ���ڽ������ݼ����ز��� -------" << endl;

    const char *pre_path = "";
    string kDataRoot = pre_path;
    kDataRoot.append("D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\data\\MNIST\\raw");
    // ��MNIST���ݽ���Normalize��Stack�������Tensor stack��һ��Tensor)
    auto test_dataset = torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), model::batch_size);

    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- ���ݼ��Ѽ��أ��ϼƻ���" << (etime - stime) << "�� -------" << endl;

    for (const auto& batch : *test_loader) {
        // ��ȡ���ݺ�label
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        // ģ��ǰ��������õ�Ԥ�����
        auto output = mnist_cnn->forward(data);
        // �������ʱ�� loss
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
    cout << "------- �����ѽ������ϼƻ���" << (test_etime - test_stime) << "�� -------" << endl;
}

void model::save_model(MNIST_CNN& net) {
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ���ڽ���ģ�ͱ������ -------" << endl;
    const char *pre_path = "D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\save_model\\";
    // ģ�ʹ��·��
    string model_save_path = pre_path;
    model_save_path.append("MNIST_CNN.pt");
    torch::save(net, model_save_path);
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- ģ���ѱ��棬�ϼƻ���" << (etime - stime) << "�� -------" << endl;
}

torch::serialize::InputArchive model::load_model() {
    auto stime = system_clock::now().time_since_epoch()/1.0s;
    cout << "------- ���ڽ���ģ�ͼ��ز��� -------" << endl;
    const char *pre_path = "";
    // ģ�ʹ��·��
    string model_load_path = pre_path;
    // ģ�ͼ���·��
    model_load_path.append("D:\\environment\\clion\\projects\\MNIST-Libtorch-cpp\\save_model\\MNIST_CNN.pt");
    torch::serialize::InputArchive archive;
    archive.load_from(model_load_path);
    auto etime = system_clock::now().time_since_epoch()/1.0s;
    cout << " ------- ģ���Ѽ��أ��ϼƻ���" << (etime - stime) << "�� -------" << endl;
    return archive;
}