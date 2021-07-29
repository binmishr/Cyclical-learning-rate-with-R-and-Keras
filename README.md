# Cyclical-learning-rate-with-R-and-Keras

Efficientnet with R and Tf2

In this blog post I will share a way to perform cyclical learning rate, with R. I worked on top of some source code I found on a other blog, by chance, but I adjusted things to make it more similar to the fast.ai approach. Also, my blog is on R-bloggers, so other R users that might want to use cyclical learning rate with R will have less trouble to find it. Sometimes things are possible in R, but, since our community is smaller, we don’t have that many resources or tutorials compared to the python community.

What is cyclical learning rate ? In a nutshell it is mostly about varying the learning rate around a min and max value during an epoch. The interests are that : 1) you don’t need to keep trying different learning rate, 2) it works as a form of regularization. Also, it trains the network faster (a phenomenon named “super convergence”).
About the data

I wrote this code in the first place in the context of the Cassava Leaf Disease Classification, a Kaggle’s competition where the goal was to train a model to identify the disease on leafs of cassava. Like the last time the I will use an Efficientnet0.

#reticulate::py_install(packages = "tensorflow", version = "2.3.0", pip=TRUE)
library(tidyverse)
library(tensorflow)
tf$executing_eagerly()
[1] TRUE
tensorflow::tf_version()
[1] '2.3'

Here I flex with my own version of keras. Basically, it is a fork with application wrapper for the efficient net.

Disclaimer : I did not write the code for the really handy applications wrappers. It came from this commit for which the PR is hold until the fully release of tf 2.3, as stated in this PR. I am not sure why the PR is closed.

devtools::install_github("Cdk29/keras", dependencies = FALSE)
library(keras)
labels<-read_csv('train.csv')
head(labels)
# A tibble: 6 x 2
  image_id       label
  <chr>          <dbl>
1 1000015157.jpg     0
2 1000201771.jpg     3
3 100042118.jpg      1
4 1000723321.jpg     1
5 1000812911.jpg     3
6 1000837476.jpg     3
levels(as.factor(labels$label))
[1] "0" "1" "2" "3" "4"
idx0<-which(labels$label==0)
idx1<-which(labels$label==1)
idx2<-which(labels$label==2)
idx3<-which(labels$label==3)
idx4<-which(labels$label==4)
labels$CBB<-0
labels$CBSD<-0
labels$CGM<-0
labels$CMD<-0
labels$Healthy<-0
labels$CBB[idx0]<-1
labels$CBSD[idx1]<-1
labels$CGM[idx2]<-1
labels$CMD[idx3]<-1

“Would it have been easier to create a function to convert the labelling ?” You may ask.

labels$Healthy[idx4]<-1

Probably.

#labels$label<-NULL
head(labels)
# A tibble: 6 x 7
  image_id       label   CBB  CBSD   CGM   CMD Healthy
  <chr>          <dbl> <dbl> <dbl> <dbl> <dbl>   <dbl>
1 1000015157.jpg     0     1     0     0     0       0
2 1000201771.jpg     3     0     0     0     1       0
3 100042118.jpg      1     0     1     0     0       0
4 1000723321.jpg     1     0     1     0     0       0
5 1000812911.jpg     3     0     0     0     1       0
6 1000837476.jpg     3     0     0     0     1       0

Following code is retaken from this online notebook named simple-convnet, which used a better approach to create a validation set than I did in the first place (not at random, but with stratification) :

set.seed(6)

tmp = splitstackshape::stratified(labels, c('label'), 0.90, bothSets = TRUE)

train_labels = tmp[[1]]
val_labels = tmp[[2]]

#following line for knowledge distillation : 
write.csv(val_labels, file='validation_set.csv', row.names=FALSE, quote=FALSE)


train_labels$label<-NULL
val_labels$label<-NULL

head(train_labels)
         image_id CBB CBSD CGM CMD Healthy
1: 3903787097.jpg   1    0   0   0       0
2: 1026467332.jpg   1    0   0   0       0
3:  436868168.jpg   1    0   0   0       0
4: 2270851426.jpg   1    0   0   0       0
5: 3234915269.jpg   1    0   0   0       0
6: 3950368220.jpg   1    0   0   0       0
head(val_labels)
         image_id CBB CBSD CGM CMD Healthy
1: 1003442061.jpg   0    0   0   0       1
2: 1004672608.jpg   0    0   0   1       0
3: 1007891044.jpg   0    0   0   1       0
4: 1009845426.jpg   0    0   0   1       0
5: 1010648150.jpg   0    0   0   1       0
6: 1011139244.jpg   0    0   0   1       0
summary(train_labels)
   image_id              CBB               CBSD       
 Length:19256       Min.   :0.00000   Min.   :0.0000  
 Class :character   1st Qu.:0.00000   1st Qu.:0.0000  
 Mode  :character   Median :0.00000   Median :0.0000  
                    Mean   :0.05079   Mean   :0.1023  
                    3rd Qu.:0.00000   3rd Qu.:0.0000  
                    Max.   :1.00000   Max.   :1.0000  
      CGM              CMD           Healthy      
 Min.   :0.0000   Min.   :0.000   Min.   :0.0000  
 1st Qu.:0.0000   1st Qu.:0.000   1st Qu.:0.0000  
 Median :0.0000   Median :1.000   Median :0.0000  
 Mean   :0.1115   Mean   :0.615   Mean   :0.1204  
 3rd Qu.:0.0000   3rd Qu.:1.000   3rd Qu.:0.0000  
 Max.   :1.0000   Max.   :1.000   Max.   :1.0000  
summary(val_labels)
   image_id              CBB               CBSD       
 Length:2141        Min.   :0.00000   Min.   :0.0000  
 Class :character   1st Qu.:0.00000   1st Qu.:0.0000  
 Mode  :character   Median :0.00000   Median :0.0000  
                    Mean   :0.05091   Mean   :0.1023  
                    3rd Qu.:0.00000   3rd Qu.:0.0000  
                    Max.   :1.00000   Max.   :1.0000  
      CGM              CMD            Healthy      
 Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
 1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.0000  
 Median :0.0000   Median :1.0000   Median :0.0000  
 Mean   :0.1116   Mean   :0.6147   Mean   :0.1205  
 3rd Qu.:0.0000   3rd Qu.:1.0000   3rd Qu.:0.0000  
 Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
image_path<-'cassava-leaf-disease-classification/train_images/'
#data augmentation
datagen <- image_data_generator(
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.5,
  horizontal_flip = TRUE,
  fill_mode = "reflect"
)
img_path<-"cassava-leaf-disease-classification/train_images/1000015157.jpg"

img <- image_load(img_path, target_size = c(448, 448))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 448, 448, 3))
img_array<-img_array/255
# Generated that will flow augmented images
augmentation_generator <- flow_images_from_data(
  img_array, 
  generator = datagen, 
  batch_size = 1 
)
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}

par(op)

Maybe you can skip the conversion of the label into 1 and 0 and directly create train generator from the original label column of the dataframe.

train_generator <- flow_images_from_dataframe(dataframe = train_labels, 
                                              directory = image_path,
                                              generator = datagen,
                                              class_mode = "other",
                                              x_col = "image_id",
                                              y_col = c("CBB","CBSD", "CGM", "CMD", "Healthy"),
                                              target_size = c(448, 448),
                                              batch_size=16)

validation_generator <- flow_images_from_dataframe(dataframe = val_labels, 
                                              directory = image_path,
                                              class_mode = "other",
                                              x_col = "image_id",
                                              y_col = c("CBB","CBSD", "CGM", "CMD", "Healthy"),
                                              target_size = c(448, 448),
                                              batch_size=16)

About tf hub

I tried a lot of things with tf hub before using the application wrappers. The application wrappers is handy, tf hub is not (for this task.) That will be the subject of an other blog post I think.

conv_base<-keras::application_efficientnet_b0(weights = "imagenet", include_top = FALSE, input_shape = c(448, 448, 3))

freeze_weights(conv_base)

model <- keras_model_sequential() %>%
    conv_base %>% 
    layer_global_max_pooling_2d() %>% 
    layer_batch_normalization() %>% 
    layer_dropout(rate=0.5) %>%
    layer_dense(units=5, activation="softmax")

summary(model)
Model: "sequential_1"
______________________________________________________________________
Layer (type)                   Output Shape                Param #    
======================================================================
efficientnetb0 (Functional)    (None, 14, 14, 1280)        4049571    
______________________________________________________________________
global_max_pooling2d_1 (Global (None, 1280)                0          
______________________________________________________________________
batch_normalization_1 (BatchNo (None, 1280)                5120       
______________________________________________________________________
dropout_1 (Dropout)            (None, 1280)                0          
______________________________________________________________________
dense_1 (Dense)                (None, 5)                   6405       
======================================================================
Total params: 4,061,096
Trainable params: 8,965
Non-trainable params: 4,052,131
______________________________________________________________________

Cyclical learning rate

A lot of the code below came from the blog “the cool data”. The idea to have a tail and the notion of annihilation of gradient originate from this blog post on The 1cycle policy and is quite similar to the one used in fastai. The big difference is that I do not want to add an other vector of an even lower learning rate at the end of the one generated by the function Cyclic_lr, it would force me to take it into account and create an other number of iteration for the compilation of the model. I prefer the approach of dividing more and more the last element of the cycle.

callback_lr_init <- function(logs){
      iter <<- 0
      lr_hist <<- c()
      iter_hist <<- c()
}
callback_lr_set <- function(batch, logs){
      iter <<- iter + 1
      LR <- l_rate[iter] # if number of iterations > l_rate values, make LR constant to last value
      if(is.na(LR)) LR <- l_rate[length(l_rate)]
      k_set_value(model$optimizer$lr, LR)
}

callback_lr <- callback_lambda(on_train_begin=callback_lr_init, on_batch_begin=callback_lr_set)
####################
Cyclic_LR <- function(iteration=1:32000, base_lr=1e-5, max_lr=1e-3, step_size=2000, mode='triangular', gamma=1, scale_fn=NULL, scale_mode='cycle')# This callback implements a cyclical learning rate policy (CLR). # The method cycles the learning rate between two boundaries with # some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186). # The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis. # This class has three built-in policies, as put forth in the paper. # - "triangular": A basic triangular cycle w/ no amplitude scaling. # - "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle. # - "exp_range": A cycle that scales initial amplitude by gamma**(cycle iterations) at each cycle iteration. # - "sinus": A sinusoidal form cycle # # Example # > clr <- Cyclic_LR(base_lr=0.001, max_lr=0.006, step_size=2000, mode='triangular', num_iterations=20000) # > plot(clr, cex=0.2)
 
      # Class also supports custom scaling functions with function output max value of 1:
      # > clr_fn <- function(x) 1/x # > clr <- Cyclic_LR(base_lr=0.001, max_lr=0.006, step_size=400, # scale_fn=clr_fn, scale_mode='cycle', num_iterations=20000) # > plot(clr, cex=0.2)
 
      # # Arguments
      #   iteration:
      #       if is a number:
      #           id of the iteration where: max iteration = epochs * (samples/batch)
      #       if "iteration" is a vector i.e.: iteration=1:10000:
      #           returns the whole sequence of lr as a vector
      #   base_lr: initial learning rate which is the
      #       lower boundary in the cycle.
      #   max_lr: upper boundary in the cycle. Functionally,
      #       it defines the cycle amplitude (max_lr - base_lr).
      #       The lr at any cycle is the sum of base_lr
      #       and some scaling of the amplitude; therefore 
      #       max_lr may not actually be reached depending on
      #       scaling function.
      #   step_size: number of training iterations per
      #       half cycle. Authors suggest setting step_size
      #       2-8 x training iterations in epoch.
      #   mode: one of {triangular, triangular2, exp_range, sinus}.
      #       Default 'triangular'.
      #       Values correspond to policies detailed above.
      #       If scale_fn is not None, this argument is ignored.
      #   gamma: constant in 'exp_range' scaling function:
      #       gamma**(cycle iterations)
      #   scale_fn: Custom scaling policy defined by a single
      #       argument lambda function, where 
      #       0 <= scale_fn(x) <= 1 for all x >= 0.
      #       mode paramater is ignored 
      #   scale_mode: {'cycle', 'iterations'}.
      #       Defines whether scale_fn is evaluated on 
      #       cycle number or cycle iterations (training
      #       iterations since start of cycle). Default is 'cycle'.
 
      ########
      if(is.null(scale_fn)==TRUE){
            if(mode=='triangular'){scale_fn <- function(x) 1; scale_mode <- 'cycle';}
            if(mode=='triangular2'){scale_fn <- function(x) 1/(2^(x-1)); scale_mode <- 'cycle';}
            if(mode=='exp_range'){scale_fn <- function(x) gamma^(x); scale_mode <- 'iterations';}
            if(mode=='sinus'){scale_fn <- function(x) 0.5*(1+sin(x*pi/2)); scale_mode <- 'cycle';}
            if(mode=='halfcosine'){scale_fn <- function(x) 0.5*(1+cos(x*pi)^2); scale_mode <- 'cycle';}
      }
      lr <- list()
      if(is.vector(iteration)==TRUE){
            for(iter in iteration){
                  cycle <- floor(1 + (iter / (2*step_size)))
                  x2 <- abs(iter/step_size-2 * cycle+1)
                  if(scale_mode=='cycle') x <- cycle
                  if(scale_mode=='iterations') x <- iter
                  lr[[iter]] <- base_lr + (max_lr-base_lr) * max(0,(1-x2)) * scale_fn(x)
            }
      }
      lr <- do.call("rbind",lr)
      return(as.vector(lr))
}

The tail

Okay, what is going on here ? Simple speaking I want the last steps to go several order of magnitude under the minimal learning rate, in a similar fashion of the fast.ai implementation. The most elegant way (without adding an other vector at the end) to do this is to divide the learning rate of the last steps by a number growing exponentially (to avoid a cut in the learning rate curve by dividing the number suddenly by 10). So we have a nice “tail” (see graphs below).

Oh there is no specific justifications for the exponent number. Just trial and error and “looking nice” approach.

n=200
nb_epochs=10
tail <- 30 #annhilation of the gradient
i<-1:tail
l_rate_div<-1.1*(1.2^i) 
plot(l_rate_div, type="b", pch=16, cex=0.1, xlab="iteration", ylab="learning rate dividor")

l_rate_cyclical <- Cyclic_LR(iteration=1:n, base_lr=1e-7, max_lr=1e-3, step_size=floor(n/2),
                        mode='triangular', gamma=1, scale_fn=NULL, scale_mode='cycle')

start_tail <-length(l_rate_cyclical)-tail
end_tail <- length(l_rate_cyclical)
l_rate_cyclical[start_tail:end_tail] <- l_rate_cyclical[start_tail:end_tail]/l_rate_div
l_rate <- rep(l_rate_cyclical, nb_epochs)

plot(l_rate_cyclical, type="b", pch=16, xlab="iteration", cex=0.2, ylab="learning rate", col="grey50")

plot(l_rate, type="b", pch=16, xlab="iteration", cex=0.2, ylab="learning rate", col="grey50")

model %>% compile(
    optimizer=optimizer_rmsprop(lr=1e-5),
    loss="categorical_crossentropy",
    metrics = "categorical_accuracy"
)

You can still add other callback, the following code came from the tutorial of Keras “tutorial_save_and_restore”. Commented to lighten the blog post.

# checkpoint_dir <- "checkpoints"
# unlink(checkpoint_dir, recursive = TRUE)
# dir.create(checkpoint_dir)
# filepath <- file.path(checkpoint_dir, "eff_net_weights.{epoch:02d}.hdf5")
# check_point_callback <- callback_model_checkpoint(
#   filepath = filepath,
#   save_weights_only = TRUE,
#   save_best_only = TRUE
# )
#callback_list<-list(callback_lr, check_point_callback ) #callback to update lr
callback_list<-list(callback_lr)
history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch=n,
    epochs = nb_epochs,
    callbacks = callback_list, #callback to update cylic lr
    validation_data = validation_generator,
    validation_step=40
)
plot(history)

Fine tuning

Here the steps, are, basically the same, you you want to divide the maximum learning rate by 5 or 10, since you unfreeze the basal part of the network.

unfreeze_weights(conv_base, from = 'block5a_expand_conv')
summary(model)
Model: "sequential_1"
______________________________________________________________________
Layer (type)                   Output Shape                Param #    
======================================================================
efficientnetb0 (Functional)    (None, 14, 14, 1280)        4049571    
______________________________________________________________________
global_max_pooling2d_1 (Global (None, 1280)                0          
______________________________________________________________________
batch_normalization_1 (BatchNo (None, 1280)                5120       
______________________________________________________________________
dropout_1 (Dropout)            (None, 1280)                0          
______________________________________________________________________
dense_1 (Dense)                (None, 5)                   6405       
======================================================================
Total params: 4,061,096
Trainable params: 3,707,853
Non-trainable params: 353,243
______________________________________________________________________
nb_epochs<-20
l_rate_cyclical <- Cyclic_LR(iteration=1:n, base_lr=1e-7, max_lr=(1e-3/5), step_size=floor(n/2),
                        mode='triangular', gamma=1, scale_fn=NULL, scale_mode='cycle')
start_tail <-length(l_rate_cyclical)-tail
end_tail <- length(l_rate_cyclical)
l_rate_cyclical[start_tail:end_tail] <- l_rate_cyclical[start_tail:end_tail]/l_rate_div

l_rate <- rep(l_rate_cyclical, nb_epochs)

#plot(l_rate, type="b", pch=16, xlab="iteration", cex=0.2, ylab="learning rate", col="grey50")
model %>% compile(
    optimizer=optimizer_rmsprop(lr=1e-5),
    loss="categorical_crossentropy",
    metrics = "categorical_accuracy"
)
callback_list<-list(callback_lr)
history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch=n,
    epochs = nb_epochs,
    callbacks = callback_list, #callback to update cylic lr
    validation_data = validation_generator,
    validation_step=40
)
plot(history)

Conclusion

And this is how you (can) do cyclical learning rate with R.

