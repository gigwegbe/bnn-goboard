// bnn_classifier.v - one-layer binary neural network classifier
// Reproduces the golden_reference.py computation in hardware:
//   for each class: accumulate popcount(XNOR(weights, image)) over 16 chunks
//   then output the class with the highest accumulated score (argmax).
//
// The weight ROM (160 words) is baked in from weights.mem.
// The image arrives as a 256-bit bus: chunk c = img_flat[16*c +: 16].

module bnn_classifier (
    input  wire         clk,
    input  wire         rst,        // synchronous, active high
    input  wire         start,      // pulse high for one cycle to begin
    input  wire [255:0] img_flat,   // 16 chunks x 16 bits
    output reg          done,       // high when a result is ready
    output reg  [3:0]   pred        // predicted digit 0..9
);
    // -------- popcount: number of 1 bits in a 16-bit value --------
    function [4:0] popcount16(input [15:0] x);
        integer i;
        begin
            popcount16 = 5'd0;
            for (i = 0; i < 16; i = i + 1)
                popcount16 = popcount16 + x[i];
        end
    endfunction

    // -------- weight ROM: 10 classes x 16 cycles = 160 words --------
    reg [15:0] wrom [0:159];
    initial $readmemh("weights.mem", wrom);

    // -------- state machine --------
    localparam IDLE = 2'd0, RUN = 2'd1, FIN = 2'd2;
    reg [1:0] state;
    reg [3:0] cls;          // current class 0..9
    reg [3:0] cyc;          // current chunk 0..15
    reg [8:0] acc;          // running score for this class (0..256)
    reg [8:0] best;         // best score seen so far
    reg [3:0] best_idx;     // class index of best score

    wire [7:0]  w_addr   = cls * 16 + cyc;
    wire [15:0] w_word   = wrom[w_addr];
    wire [15:0] img_word = img_flat[16*cyc +: 16];
    wire [15:0] agree    = ~(w_word ^ img_word);            // XNOR
    wire [4:0]  pc       = popcount16(agree);
    wire [8:0]  running  = (cyc == 4'd0) ? {4'b0, pc}       // first chunk
                                         : acc + pc;        // accumulate

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; done <= 1'b0; pred <= 4'd0;
            cls <= 0; cyc <= 0; acc <= 0; best <= 0; best_idx <= 0;
        end else case (state)
            IDLE: if (start) begin
                state <= RUN; done <= 1'b0;
                cls <= 0; cyc <= 0; acc <= 0; best <= 0; best_idx <= 0;
            end

            RUN: begin
                if (cyc == 4'd15) begin
                    // class finished; 'running' holds its full 256-bit score
                    if (running > best) begin
                        best     <= running;
                        best_idx <= cls;
                    end
                    cyc <= 0; acc <= 0;
                    if (cls == 4'd9) state <= FIN;
                    else             cls <= cls + 4'd1;
                end else begin
                    acc <= running;          // carry score into next chunk
                    cyc <= cyc + 4'd1;
                end
            end

            FIN: begin
                pred <= best_idx;
                done <= 1'b1;
                if (start) begin             // allow a re-run
                    state <= RUN; done <= 1'b0;
                    cls <= 0; cyc <= 0; acc <= 0; best <= 0; best_idx <= 0;
                end
            end
        endcase
    end
endmodule
