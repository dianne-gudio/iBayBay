package com.example.ibaybayversion1;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class OpenGalleryActivity extends AppCompatActivity {
    private TextView yourImageText;
    private ImageView image;
    private Button selectBtn;
    Bitmap img;
    String encoded;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_open_gallery);

        //Assign variables
        yourImageText = findViewById(R.id.yourImageText);
        image = findViewById(R.id.selectedImage);
        selectBtn = (Button) findViewById(R.id.selectBtn);

        //"Select Image" Button
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 100){
            //Get the image that is selected
            Uri uri = data.getData();
            image.setImageURI(uri);

            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            //-----------------NOW ANALYZE BUTTON-----------------
            selectBtn.setText("Analyze");
            selectBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    connectPython();
                }
            });
        }
        yourImageText.setVisibility(View.INVISIBLE); //Hides "Your captured image will show here." message

        //Converts Bitmap image to base 64 string
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        img.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
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