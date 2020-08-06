#include <stdio.h>
#include "Gap.h"
#include "layer_params.h"
#include "FireGenerator.h"

static char *Str(char *S, int A)
{
    return S;
}

void Fire(const char *Name, CNN_GenControl_T *Ctrl, unsigned idx)
{
#ifdef GRAPH
    int w_InL3 = 0, b_InL3 = 0;
#else
    int w_InL3 = 1, b_InL3 = 1;
#endif

#ifdef GRAPH
    OpenKernelGroup(Name);
#endif
    for (unsigned i = idx; i < idx + 3; i++)
    {
        CNN_ConvolutionPoolReLU(
            convLayers[i].name,
            Ctrl,
            2,2,2,2,             // All short ints
            convLayers[i].q.in,  // Input quantization
            convLayers[i].q.weights, // Weight quantization
            convLayers[i].q.bias,// Bias quantization
            convLayers[i].q.out, // Output quantization
            0,w_InL3,b_InL3,0,
            convLayers[i].nb_if, // InFeat
            convLayers[i].nb_of, // OutFeat
            convLayers[i].win,   // Width
            convLayers[i].hin,   // Height
            KOP_CONV,
            convLayers[i].kernel_width,  // FScW
            convLayers[i].kernel_height, // FScH
            1,
            1,
            convLayers[i].conv_stride, // ConvStrideW
            convLayers[i].conv_stride, // ConvStrideH
            convLayers[i].conv_padding,

            convLayers[i].max_pool ? KOP_MAXPOOL : KOP_NONE, //Max Pool
            convLayers[i].pool_size, // FSpW
            convLayers[i].pool_size, // FSpH
            1,  //Dilation x
            1,  //Dilation y
            convLayers[i].pool_stride, // PoolStrideW
            convLayers[i].pool_stride, // PoolStrideH
            0, // PoolDoPad

            convLayers[i].relu ? KOP_RELU : KOP_NONE
        );
    }

#ifdef GRAPH
    CloseKernelGroup();
#endif

#ifdef GRAPH
    char S0[256], S1[256], S2[256];
    int A;
    char *DataType = CNN_ArgDataType(2, 1, 1);

    CKernel_Arg_T **GroupCArgs;
    StackedTensors_T *GroupStackedTensors;
    CKernelCall_T **GroupCCalls;
    Object_T **GroupKerArgs;

    GroupCArgs = AllocateCArgs(9);

    A = 0;
    GroupCArgs[A++] = TCArg(DataType, "In");

    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_W", convLayers[idx].name)));
    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_B", convLayers[idx].name)));
    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_O", convLayers[idx].name)));

    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_W", convLayers[idx+1].name)));
    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_B", convLayers[idx+1].name)));

    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_W", convLayers[idx+2].name)));
    GroupCArgs[A++] = TCArg(DataType, Str(S0, sprintf(S0,"%s_B", convLayers[idx+2].name)));

    GroupCArgs[A++] = TCArg(DataType, "Out");

    GroupStackedTensors = AT_StackedTensors("Out", 2, "Out_C1x1", "Out_C3x3");

    GroupCCalls = AllocateCalls(3);

    A = 0;
    GroupCCalls[A++] = UserKernelCall(convLayers[idx].name, LOC_GROUP,
                Bindings(4,
                    C_Arg("In"),
                    C_Arg(Str(S0, sprintf(S0,"%s_W", convLayers[idx].name))),
                    C_Arg(Str(S1, sprintf(S1,"%s_B", convLayers[idx].name))),
                    C_Arg(Str(S2, sprintf(S2,"%s_O", convLayers[idx].name)))));
    GroupCCalls[A++] = UserKernelCall(convLayers[idx+1].name, LOC_GROUP,
                Bindings(4,
                    C_Arg(Str(S0, sprintf(S0,"%s_O", convLayers[idx].name))),
                    C_Arg(Str(S1, sprintf(S1,"%s_W", convLayers[idx+1].name))),
                    C_Arg(Str(S2, sprintf(S2,"%s_B", convLayers[idx+1].name))),
                    C_Arg("Out_C1x1")));
    GroupCCalls[A++] = UserKernelCall(convLayers[idx+2].name, LOC_GROUP,
                Bindings(4,
                    C_Arg(Str(S0, sprintf(S0,"%s_O", convLayers[idx].name))),
                    C_Arg(Str(S0, sprintf(S0,"%s_W", convLayers[idx+2].name))),
                    C_Arg(Str(S1, sprintf(S1,"%s_B", convLayers[idx+2].name))),
                    C_Arg("Out_C3x3")));

    GroupKerArgs = AllocateKerArgs(9);

    A = 0;
    GroupKerArgs[A++] = KerGroupArg("In", O_IN, get_layer_in_size(idx), 2, "In");

    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_W", convLayers[idx].name)),   O_IN,       get_layer_weights_size(idx),    2, Str(S1, sprintf(S1,"%s_W", convLayers[idx].name)));
    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_B", convLayers[idx].name)),   O_IN,       get_layer_bias_size(idx),       2, Str(S1, sprintf(S1,"%s_B", convLayers[idx].name)));
    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_O", convLayers[idx].name)), O_IN|O_OUT|O_BUFF,  get_layer_out_size(idx),  2, Str(S1, sprintf(S1,"%s_O", convLayers[idx].name)));

    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_W", convLayers[idx+1].name)), O_IN,       get_layer_weights_size(idx+1),  2, Str(S1, sprintf(S1,"%s_W", convLayers[idx+1].name)));
    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_B", convLayers[idx+1].name)), O_IN,       get_layer_bias_size(idx+1),     2, Str(S1, sprintf(S1,"%s_B", convLayers[idx+1].name)));

    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_W", convLayers[idx+2].name)), O_IN,       get_layer_weights_size(idx+2),  2, Str(S1, sprintf(S1,"%s_W", convLayers[idx+2].name)));
    GroupKerArgs[A++] = KerGroupArg(Str(S0, sprintf(S0,"%s_B", convLayers[idx+2].name)), O_IN,       get_layer_bias_size(idx+2),     2, Str(S1, sprintf(S1,"%s_B", convLayers[idx+2].name)));

    GroupKerArgs[A++] = KerGroupArg("Out", O_OUT, get_layer_out_size(idx+1) + get_layer_out_size(idx+2), 2, "Out");

    UserKernelGroupK(
        Name,
        1,
        GroupCArgs,
        GroupStackedTensors,
        GroupCCalls,
        GroupKerArgs
    );
#endif // GRAPH
}
