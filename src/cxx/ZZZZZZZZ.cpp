
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <iron.h>

int main(int argc, char *argv[])
{
    std::ofstream of;
    std::vector<std::string> program_args(argv, argv + argc);
    
    of.open("results.txt");
    
    if (argc > 2)
    {
        if (program_args[1].compare("typeA") == 0 && program_args[2].compare("partB") == 0)\
        {
            of << "correct." << std::endl;
        }
        else
        {
            of << "almost." << std::endl;
        }
    }
    else 
    {
        of << "wrong." << std::endl;
    }
    
    of.close();
    
    return 0;
}