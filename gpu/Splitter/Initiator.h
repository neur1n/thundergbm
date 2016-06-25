/*
 * DevicePredKernel.h
 *
 *  Created on: 21 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDKERNEL_H_
#define DEVICEPREDKERNEL_H_

#include "../../DeviceHost/DefineConst.h"
#include "../../DeviceHost/TreeNode.h"
#include "../../pureHost/UpdateOps/NodeStat.h"

__global__ void ComputeGDKernel(int numofIns, float_point *pfPredValue, float_point *pfTrueValue, float_point *pGrad, float_point *pHess);
__global__ void InitNodeStat(int numofIns, float_point *pGrad, float_point *pHess,
							 nodeStat *pSNodeStat, int *pSNIdToBuffId, int maxNumofSplittable, int *pBuffId);
__global__ void InitRootNode(TreeNode *pAllTreeNode, int *pCurNumofNode);

#endif /* DEVICEPREDKERNEL_H_ */
