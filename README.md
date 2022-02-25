## resource_packing_self_play

This is the code of our paper "Self-play learning strategies for resource assignment in Open-RAN networks".

Please use different branches for training, testing (validation), and comparison with heuristic methods.

`bin_packing_various_height`: training.

`validate_bin_packing`: validation of trained models.

`vanilla-MCTS`: vanilla MCTS method. This branch also contains a file for generating .gif files of bin packing results (for demo).

`HVRAA-heuristics`: heuristic method.

`lego-heuristic`: heuristic method.

----Please ignore----

`master`: not in use.

`bin_packing_single_height`: training using single height samples. This is not what we did in the paper.

```
@article{wang2022self,  
  title={Self-play learning strategies for resource assignment in Open-RAN networks},  
  author={Wang, Xiaoyang and Thomas, Jonathan D and Piechocki, Robert J and Kapoor, Shipra and Santos-Rodr{\'\i}guez, Ra{\'u}l and Parekh, Arjun},  
  journal={Computer Networks},  
  pages={108682},  
  year={2022},  
  publisher={Elsevier}  
}
```
