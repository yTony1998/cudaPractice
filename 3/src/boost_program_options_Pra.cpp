#include <iostream>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

int main(int argc, char const *argv[])
{
    int apple_num = 0, orange_num = 0;
    std::vector<std::string> addr;
    boost::program_options::options_description opt("all options");

    opt.add_options()
    ("apple,a", boost::program_options::value<int>(&apple_num)->default_value(10), "苹果数量")
    ("orange,o", boost::program_options::value<int>(&orange_num)->default_value(20), "橘子数量")
    ("address",boost::program_options::value<std::vector<std::string>>()->multitoken(),"生产地")
    ("help", "计算苹果和橘子的数量");

    boost::program_options::variables_map vm;

    try
    {
        boost::program_options::store (parse_command_line(argc, argv, opt), vm);
    }
    catch(...)
    {

        std::cout<<"输入的参数中存在未定义的选项！\n";
        return 0;
    }
    boost::program_options::notify(vm);

    if(vm.count("help"))
    {
        std::cout<<opt<<std::endl;
        return 0;
    }
    if(vm.count("address"))
    {
        std::cout<<"生产地为：";
        for (auto& str : vm["address"].as<std::vector<std::string>>())
        {
            std::cout << str <<" ";
        }
        std::cout<<std::endl;
        
    }
    std::cout <<"苹果的数量："<< apple_num << std::endl;
    std::cout <<"橘子的数量："<< orange_num << std::endl;
    std::cout <<"总数量：" << apple_num + orange_num << std::endl;
    return 0;


}