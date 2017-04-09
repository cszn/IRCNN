
## Learning Deep CNN Denoiser Prior for Image Restoration (CVPR 2017)

### Abstract:

```
Model-based optimization methods and discriminative learning methods have been the two dominant strategies 
for solving various inverse problems in low-level vision.Typically, those two kinds of methods have their
respective merits and drawbacks, e.g., model-based optimization methods are flexible for handling different
inverse problems but are usually time-consuming with sophisticated priors for the purpose of good performance; 
in the meanwhile, discriminative learning methods have fast testing speed but their application range is greatly
restricted by the specialized task.Recent works have revealed that, with the aid of variable splitting techniques, 
denoiser prior can be plugged in as a modular part of model-based optimization methods to solve other inverse 
problems (e.g., deblurring). Such an integration induces considerable advantage when the denoiser is obtained 
via discriminative learning. However, the study of integration with fast discriminative denoiser prior is still 
lacking. To this end, this paper aims to train a set of fast and effective CNN (convolutional neural network) 
denoisers and integrate them into model-based optimization method to solve other inverse problems. Experimental
results demonstrate that the learned set of denoisers can not only achieve promising Gaussian denoising results 
but also can be used as prior to deliver good performance for various low-level vision applications.
```




To run the matlab code, you should install the [MatConvNet](http://www.vlfeat.org/matconvnet/) first.
```
 @inproceedings{zhang2017learning,
   title={Learning Deep CNN Denoiser Prior for Image Restoration},
   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2017},
 }
 ```
