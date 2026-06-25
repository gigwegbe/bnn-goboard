// top.v - synthesizable wrapper for the Nandland Go Board (iCE40-HX1K).
// Bakes in one selected image, runs the classifier once at power-up, and
// shows the predicted digit on the 4 LEDs (binary) and the 7-seg display.
//
// STAGE 3 TODO: pin names/numbers (.pcf) and segment polarity are board
// specifics we finalize next. The logic below is complete and simulated.

module top #(
    parameter IMG_ID = 4              // which stored image to classify (0..9)
) (
    input  wire i_clk,                // 25 MHz Go Board clock
    output wire [3:0] o_led,          // 4 user LEDs = predicted digit in binary
    output wire [6:0] o_seg           // 7-seg segments (active-low on real board)
);
    // --- baked image data (pre-transposed), selects one image ---
    reg [15:0] imgmem [0:159];
    initial $readmemh("images.mem", imgmem);

    reg [255:0] img_flat;
    integer c;
    always @(*) begin
        for (c = 0; c < 16; c = c + 1)
            img_flat[16*c +: 16] = imgmem[IMG_ID*16 + c];
    end

    // --- power-on reset then a single start pulse ---
    reg [4:0] por = 5'd0;
    always @(posedge i_clk) if (por != 5'h1f) por <= por + 5'd1;
    wire rst   = (por < 5'd4);
    wire start = (por == 5'd8);

    // --- classifier ---
    wire        done;
    wire [3:0]  pred;
    bnn_classifier u_bnn (
        .clk(i_clk), .rst(rst), .start(start),
        .img_flat(img_flat), .done(done), .pred(pred)
    );

    // --- outputs ---
    assign o_led = pred;

    wire [6:0] seg_ah;
    seg7 u_seg (.digit(pred), .seg(seg_ah));
    assign o_seg = ~seg_ah;          // Go Board segments are active-low
endmodule
