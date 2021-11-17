package com.example.ibaybayversion1;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

public class CaptureImageActivity extends AppCompatActivity {
    private ImageView capturedImage;
    private Button captureBtn;
    private TextView yourImageText;
    int requestCode = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_capture_image);

        //Assign variables
        capturedImage = findViewById(R.id.capturedImage);
        captureBtn = (Button) findViewById(R.id.captureBtn);
        yourImageText = findViewById(R.id.yourImageText);

        //Request for camera permission
        if (ContextCompat.checkSelfPermission(CaptureImageActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
        {
            ActivityCompat.requestPermissions(CaptureImageActivity.this, new String[]{
                            Manifest.permission.CAMERA}, requestCode);
        }

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(i, requestCode);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (requestCode == 100){
            //Get captured image
            Bitmap finalImage = (Bitmap) data.getExtras().get("data");

            //Display the captured image to Image View
            capturedImage.setImageBitmap(finalImage);

            //-----------------ANALYZE IMAGE-----------------TO BE EDITED---------
            captureBtn.setText("Analyze");
            captureBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showOutput();
                }
            });
        }
        yourImageText.setVisibility(View.INVISIBLE); //Hides "Your captured image will show here." message
    }

    public void showOutput(){
        Intent i = new Intent(this, OutputActivity.class); //Opens Output screen
        startActivity(i);
    }
}