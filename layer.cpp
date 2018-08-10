#include "layer.h"

Layer::Layer() {
	mLearningRate = 0.01;
}


LayerType Layer::type() {
	return mLayerType;
}