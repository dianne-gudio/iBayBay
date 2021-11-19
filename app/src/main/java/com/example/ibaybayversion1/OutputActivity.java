package com.example.ibaybayversion1;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.FileUtils;
import android.provider.MediaStore;
import android.provider.Settings;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.IOException;
import java.util.Base64;
import java.util.List;

public class OutputActivity extends AppCompatActivity {
    public static ImageView outputImage;
    public static TextView outputResult;
    Button doneBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_output);

        //Assign variables
        doneBtn = (Button) findViewById(R.id.doneBtn);
        outputResult = (TextView) findViewById(R.id.outputResult);
        outputImage = findViewById(R.id.outputImage);

        // Output
        outputResult.setText(GlobalVariables.classification);
        outputImage.setImageURI(GlobalVariables.imageInput);

        //"Done" Button
        doneBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                backToHome();
            }
        });
    }

    public void backToHome(){
        Intent i = new Intent(this, MainActivity.class); //Opens Main Menu screen
        startActivity(i);
    }
}