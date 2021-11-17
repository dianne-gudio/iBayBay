package com.example.ibaybayversion1;

import android.content.Intent;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
    private Button captureBtn;
    private Button chartBtn;
    private Button uploadBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Assign variables
        captureBtn = (Button) findViewById(R.id.selectBtn);
        chartBtn = (Button) findViewById(R.id.chartBtn);
        uploadBtn = (Button) findViewById(R.id.uploadBtn);

        //"Capture Image" button
        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
            }
        });

        //"Upload Image" Button
        uploadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });

        //"Open Baybayin Chart" button
        chartBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openChart();
            }
        });
    }

    public void openCamera(){
        Intent i = new Intent(this, CaptureImageActivity.class); //Opens the screen for Capture Image
        startActivity(i);
    }

    public void openGallery() {
        Intent i = new Intent(this, OpenGalleryActivity.class); //Opens the screen for Upload Image
        startActivity(i);
    }

    public void openChart(){
        Intent i = new Intent(this, OpenChartActivity.class); //Opens the screen for Baybayin Chart
        startActivity(i);
    }
}