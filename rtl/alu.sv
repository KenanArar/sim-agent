// 8-bit ALU — demo DUT for the L4 RTL verification agent.
//
// This file has two planted bugs so the cocotb regression has real failures
// to cluster. They're deliberately unmarked here so the agent has to find them
// from the failure data (answers: README "Level 4" / tests/rtl_smoke_test.py).
module alu (
    input  logic [7:0] a,
    input  logic [7:0] b,
    input  logic [2:0] op,
    output logic [7:0] y,
    output logic       zero
);
    localparam logic [2:0] OP_ADD = 3'd0,
                           OP_SUB = 3'd1,
                           OP_AND = 3'd2,
                           OP_OR  = 3'd3,
                           OP_XOR = 3'd4,
                           OP_SLT = 3'd5,
                           OP_SLL = 3'd6,
                           OP_SRL = 3'd7;

    logic [2:0] shamt;
    assign shamt = b[2:0];

    always_comb begin
        case (op)
            OP_ADD:  y = a + b;
            OP_SUB:  y = a + ~b;
            OP_AND:  y = a & b;
            OP_OR:   y = a | b;
            OP_XOR:  y = a ^ b;
            OP_SLT:  y = {7'b0, a < b};
            OP_SLL:  y = a << shamt;
            OP_SRL:  y = a >> shamt;
            default: y = 8'h00;
        endcase
    end

    assign zero = (y == 8'h00);
endmodule
