package com.example.ibaybayversion1;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.os.FileUtils;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.text.SimpleDateFormat;
import android.util.Base64;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.util.Date;
import java.util.List;

public class CaptureImageActivity extends AppCompatActivity {
    ImageView capturedImage;
    Button captureBtn;
    TextView yourImageText;
    int requestCode = 100;
    Bitmap finalImage;
    String encoded;

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
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (requestCode == 100){
            //Get captured image
            finalImage = (Bitmap) data.getExtras().get("data");

            //Display the captured image to Image View
            capturedImage.setImageBitmap(finalImage);

            //-----------------NOW ANALYZE BUTTON-----------------
            captureBtn.setText("Analyze");
            captureBtn.setOnClickListener(new View.OnClickListener() {

                @Override
                public void onClick(View v) {
                    connectPython();
                }

            });
        }
        yourImageText.setVisibility(View.INVISIBLE); //Hides "Your captured image will show here." message

        //Converts Bitmap image to base 64 string
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        finalImage.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream .toByteArray();

        encoded = Base64.encodeToString(byteArray, Base64.DEFAULT);

    }

    public void connectPython(){

        /*
        ---------------------EDIT HERE----------------------

        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();

        PyObject pyobj = py.getModule("baybayin_cnn_classify"); //Python File
        PyObject obj = pyobj.callAttr("get_data"); //Python def
        List<PyObject> result = obj.callAttr("get_data", encoded).asList(); //Not sure if this is applicable sa app natin - kasi parang iba yung structure ng code nila Raymond

        OutputActivity.outputImage.setImageBitmap(); //Displays the output image in app, still to be edited.
        OutputActivity.outputResult.setText(obj.toString()); //Displays the output prediction to app, still to be edited.

        */

        showOutput();
    }

    public void showOutput(){
        Intent i = new Intent(this, OutputActivity.class); //Opens Output screen
        startActivity(i);
    }
}