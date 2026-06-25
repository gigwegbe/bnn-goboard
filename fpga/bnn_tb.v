`timescale 1ns/1ps
// bnn_tb.v - drives all 10 stored images through the classifier and checks
// each prediction against the golden reference expectations.
module bnn_tb;
    reg clk = 0, rst = 1, start = 0;
    reg [255:0] img_flat = 0;
    wire done;
    wire [3:0] pred;

    bnn_classifier dut (.clk(clk), .rst(rst), .start(start),
                        .img_flat(img_flat), .done(done), .pred(pred));

    always #10 clk = ~clk;                 // 50 MHz simulation clock

    reg [15:0] imgs [0:159];               // 10 images x 16 chunks
    integer expected [0:9];
    integer m, c, errors;

    initial begin
        $readmemh("images.mem", imgs);
        expected[0]=0; expected[1]=1; expected[2]=2; expected[3]=3; expected[4]=9;
        expected[5]=5; expected[6]=6; expected[7]=7; expected[8]=8; expected[9]=9;
        errors = 0;

        @(negedge clk); rst = 0;

        for (m = 0; m < 10; m = m + 1) begin
            for (c = 0; c < 16; c = c + 1)
                img_flat[16*c +: 16] = imgs[m*16 + c];

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            wait (done == 1'b0);           // result clears as it starts
            wait (done == 1'b1);           // ...and is ready again
            @(negedge clk);

            if (pred === expected[m])
                $display("image %0d -> predicted %0d   OK", m, pred);
            else begin
                $display("image %0d -> predicted %0d   MISMATCH (expected %0d)",
                         m, pred, expected[m]);
                errors = errors + 1;
            end
        end

        $display("");
        if (errors == 0)
            $display("PASS: all 10 predictions match golden_reference.py");
        else
            $display("FAIL: %0d mismatch(es)", errors);
        $finish;
    end

    initial begin #5_000_000 $display("TIMEOUT"); $finish; end
endmodule
