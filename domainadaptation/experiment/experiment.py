from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from ..tester import Tester
from ..trainer import Trainer
from ..visualizer import Visualizer
from ..data_provider import DomainGenerator
from ..models import GradientReversal


class Experiment:
    class BackboneType(str, Enum):
        ALEXNET = "alexnet"
        VGG16 = "vgg16"
        RESNET50 = "resnet50"
        RESNET101 = "resnet101"

        def __str__(self):
            return self.value

    def __init__(self, config):
        self.config = config
        
        self._kwargs_for_backbone = {
            'include_top': False,
            'weights': config['backbone']['weights'],
            'input_shape': (*config['backbone']['img-size'], 3),
            'pooling': config['backbone']['pooling'],
        }

        if config["backbone"]["type"] == self.BackboneType.ALEXNET:
            raise NotImplementedError
        elif config["backbone"]["type"] == self.BackboneType.VGG16:
            self._backbone_class = keras.applications.vgg16.VGG16
            preprocess_input = keras.applications.vgg16.preprocess_input
        elif config["backbone"]["type"] == self.BackboneType.RESNET50:
            self._backbone_class = keras.applications.resnet.ResNet50
            preprocess_input = keras.applications.resnet.preprocess_input
        elif config["backbone"]["type"] == self.BackboneType.RESNET101:
            self._backbone_class = keras.applications.resnet.ResNet101
            preprocess_input = keras.applications.resnet.preprocess_input
        else:
            raise ValueError("Not supported backbone type")
            

        self.domain_generator = DomainGenerator(config["dataset"]["path"],
                                                preprocessing_function=preprocess_input,
                                                **config["dataset"]["augmentations"])
    
    def _get_new_backbone_instance(self, **kwargs):
        if kwargs:
            new_kwargs = self._kwargs_for_backbone.copy()
            new_kwargs.update(kwargs)
            return self._backbone_class(**new_kwargs)
        else:
            return self._backbone_class(**self._kwargs_for_backbone) 

    @staticmethod
    def _cross_entropy(model, x_batch, y_batch):
        logits = model(x_batch)
        return tf.nn.softmax_cross_entropy_with_logits(y_batch, logits)

    @staticmethod
    def _domain_wrapper(generator, domain=0):
        """ Changes y_batch from class labels to source/target domain label """

        assert domain in [0, 1], "wrong domain number"

        for X_batch, _ in generator:
            y_batch = np.zeros(shape=(X_batch.shape[0], 2))
            y_batch[:, domain] = 1
            yield X_batch, tf.convert_to_tensor(y_batch)

    @staticmethod
    def _get_classifier_head(num_classes):
        return keras.layers.Dense(units=num_classes)


class DANNExperiment(Experiment):
    """
    Domain-Adversarial Training of Neural Networks
    link: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, config):
        super().__init__(config)

    def experiment_no_domain_adaptation(self):
        backbone = self._get_new_backbone_instance()
        
        for i in range(len(backbone.layers)):
            backbone.layers[i].trainable = False
        
        classifier_head = self._get_classifier_head(num_classes=self.config["dataset"]["classes"])
        
        classification_model = keras.Model(
            inputs=backbone.inputs,
            outputs=classifier_head(backbone.outputs[0]))
        
        
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]
        )
        
        trainer = Trainer()
        optimizer = keras.optimizers.Adam()
        
        for i in range(self.config["epochs"]):
            trainer.train(
                model=classification_model,
                compute_loss=self._cross_entropy,
                optimizer=optimizer,
                train_generator=source_generator,
                steps=self.config["steps"],
                callbacks=[lambda **kwargs: print(tf.reduce_mean(kwargs['loss']))]
            )
            print('epoch {} finished'.format(i + 1))
        
        
        tester = Tester()
        tester.test(classification_model, source_generator)
        
        target_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]
        )
        target_tester = Tester()
        target_tester.test(classification_model, target_generator)
        
    def experiment_domain_adaptation(self):
        backbone = self._get_new_backbone_instance()
        
        for i in range(len(backbone.layers) - 5):
            backbone.layers[i].trainable = False
        
        classifier_head = keras.Model(
            inputs=backbone.inputs,
            outputs=self._get_classifier_head(num_classes=self.config["dataset"]["classes"])(backbone.outputs[0])
        )
        
        lambda_coef = tf.Variable(self._get_lambda(0.), trainable=False, dtype=tf.float32)
        domain_head = keras.Model(
            inputs=backbone.inputs,
            outputs=self._get_classifier_head(num_classes=2)(GradientReversal(lambda_coef)(backbone.outputs[0]))
        )
        
        
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]
        )
        
        target_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]
        )
        
        source_domain_generator = self._domain_wrapper(source_generator, domain=0)
        target_domain_generator = self._domain_wrapper(target_generator, domain=1)
        
        trainer = Trainer()
        classifier_head_optimizer = keras.optimizers.Adam(1e-4)
        domain_head_optimizer = keras.optimizers.Adam(1e-4)
        
        for i in range(self.config["epochs"]):
            lambda_coef.assign(self._get_lambda(i / self.config["epochs"]))
            
            trainer.train(
                model=classifier_head,
                compute_loss=self._cross_entropy,
                optimizer=classifier_head_optimizer,
                train_generator=source_generator,
                steps=len(source_generator),
                callbacks=[lambda **kwargs: print(tf.reduce_mean(kwargs['loss']))]
            )
            print('Classification epoch {} finished\n'.format(i + 1))
            
            for j in range(2 * min(len(source_generator), len(target_generator))):
                generator = source_domain_generator if j % 2 == 0 else target_domain_generator
                trainer.train(
                    model=domain_head,
                    compute_loss=self._cross_entropy,
                    optimizer=domain_head_optimizer,
                    train_generator=generator,
                    steps=1,
                    callbacks=[lambda **kwargs: print(tf.reduce_mean(kwargs['loss']))]
                )
                
            if len(source_generator) > len(target_generator):
                generator = source_domain_generator
            else:
                generator = target_domain_generator
            
            trainer.train(
                model=domain_head,
                compute_loss=self._cross_entropy,
                optimizer=domain_head_optimizer,
                train_generator=generator,
                steps=max(len(source_generator), len(target_generator))\
                    - min(len(source_generator), len(target_generator)),
                callbacks=[lambda **kwargs: print(tf.reduce_mean(kwargs['loss']))]
            )
            print('Domain classification epoch {} finished\n'.format(i + 1))
            
            tester = Tester()
            tester.test(classifier_head, source_generator)
            tester.test(classifier_head, target_generator)
        
    @staticmethod
    def _get_lambda(p=0.):
        """ Original lambda scheduler """
        return 2 / (1 + np.exp(-10 * p)) - 1
