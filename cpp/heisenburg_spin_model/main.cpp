#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Declare empty variable of integers. This will hold our states.
    std::vector<int32_t> state_vec;
    int int_type = 32;

    // Fill state_vec
    // Conservation of Magnetization:
    // The Hamilton is not creating or destroying spin, thus interaction between
    // states with different number of total up-spins will evaluate to zero

    int num_up_spin = 4; //16 ~ half up-spin
    int num_spin_chain = 8;
    int num_tot_states = std::pow(2, num_spin_chain);

    if (num_up_spin > num_spin_chain) {
        std::cerr << "Number of up-spins cannot be greater than the total number of spins in the spin chain. Duh." << std::endl;
        return 1;
    }
    if (num_spin_chain < 0; num_up_spin < 0) {
        std::cerr << "Number of up spins cannot be less than 0" << std::endl;
        return 1;
    }

    for (int i = 0; i < num_tot_states; i++) {
        // Go through each integer -> if they have num_up_spin of 1 -> add to state_vec
        int spin_up_counter = 0;
        for (int j = 0; j < int_type; j++) {
            std::cout << spin_up_counter << std::endl;
            int mask = 1 << j;
            if ((mask & i) > 0) {
                std::cout << "greater" << std::endl;
                spin_up_counter++;
                if (spin_up_counter > num_up_spin) {
                    std::cout << "breaking" << std::endl;
                    break;
                }
            }

            if (j == 31 && spin_up_counter == num_up_spin) {
                state_vec.push_back(i);
                std::cout << "pushed" << std::endl;
            } else if (j == (int_type -1) && spin_up_counter != num_up_spin) {
                std::cout << "State did not satisfy the magnetization condition." << std::endl;
            }
        }
    }

    std::cout << "Filled Array:" << state_vec.size() << std::endl;

    
    // Returning 0 tells the operating system the program finished successfully
    return 0; 
}