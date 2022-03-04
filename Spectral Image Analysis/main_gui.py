########################################################################
## IMPORTS
########################################################################
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
import numpy as np
import blend_modes
import pathlib
from ctypes import *
from PIL import Image
import io

########################################################################
# IMPORT GUI FILE
from UI.prototype6 import *
# IMPORT SEGMENTATION FILE
from segmentImagesWithAPreTrainedModel import *
from for_gui import *
########################################################################
## MAIN WINDOW CLASS
########################################################################
# sys.setrecursionlimit(2000)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # when object has class, this is way to call them from ui file:
        # self.test = self.findChild(class,"name")
        # when object don't have class, this is way to call them from ui file:
        # self.test = self.ui.name


        #######################################################################
        ## start call objects from ui file to use them in the function  
        #######################################################################

        self.method = self.findChild(QComboBox,"method_list")
        self.method.setView(QtWidgets.QListView())
        self.method.setStyleSheet("QListView::item {height:10px;}")
        self.method.currentIndexChanged.connect(self.changeMethod)

        self.run = self.findChild(QToolButton,"run_btn")
        self.run.clicked.connect(self.runNet)

        self.open = self.findChild(QToolButton,"openimg_btn")
        self.open.clicked.connect(self.clicker)
        self.rgb_img = self.findChild(QLabel,"rgb_img") # label to put opened image
        self.modified_img = self.findChild(QLabel,"modified_img") # label to put mask image

        self.rotated_angle = 0 # initiate rotation angle
        self.angle = self.findChild(QPushButton,"angle_btn")
        self.angle.clicked.connect(self.rotation)
        
        self.angle2 = self.findChild(QPushButton,"angle_btn2")
        self.angle2.clicked.connect(self.rotation2)

        self.width = 512 # initiate image width of scaling  
        self.height = 512 # initiate image height of scaling  
    
        self.up_scale = self.findChild(QPushButton,"scaleL_btn")
        self.up_scale.clicked.connect(self.up_scaling)

        self.down_scale = self.findChild(QPushButton,"scaleS_btn")
        self.down_scale.clicked.connect(self.down_scaling)
    
        self.up_scale2 = self.findChild(QPushButton,"scaleL_btn2")
        self.up_scale2.clicked.connect(self.up_scaling2)

        self.down_scale2 = self.findChild(QPushButton,"scaleS_btn2")
        self.down_scale2.clicked.connect(self.down_scaling2)

        self.mask = self.findChild(QToolButton,"openmask_btn")
        self.mask.clicked.connect(self.load_mask)

        self.slider = self.findChild(QSlider,"rate_slider")
        self.slider_label = self.findChild(QLabel,"rate_value")
        self.slider.valueChanged.connect(self.blending) 
        
        self.comboMode = self.findChild(QComboBox,"comboBox")
        self.comboMode.setView(QtWidgets.QListView())
        self.comboMode.setStyleSheet("QListView::item {height:25px;}")
        self.comboMode.currentIndexChanged.connect(self.update_blending_mode)

        self.zoom_plus = self.findChild(QPushButton,"zoomin_btn")
        self.zoom_plus.clicked.connect(self.zoomPlus)
    
        self.zoom_minus = self.findChild(QPushButton,"zoomout_btn")
        self.zoom_minus.clicked.connect(self.zoomMinus)

        self.zoom_reset = self.findChild(QPushButton,"normalsize_btn")
        self.zoom_reset.clicked.connect(self.resetZoom)

        self.zoomX = 0.5            # zoom factor for resize self.rgb_img
        self.position = [0, 0]      # position of top left corner of self.rgb_img for qimage_scaled
        self.panFlag = True         # to enable or enable pan
        self.__connectEvents()      # enable mouse events

        self.table = self.findChild(QTableWidget,"label_table")
        self.table.setSelectionBehavior(QTableWidget.SelectRows)  # Select only rows
        self.table.itemClicked.connect(self.outSelect) # connect item in table with function 


        #######################################################################
        # SHOW WINDOW
        #######################################################################
        self.show()
        ########################################################################



        ########################################################################
        ## start to define our function 
        ########################################################################


    def changeMethod(self):

        # select desired the segmentation methods
        self.selected_mode = self.method.currentText()
        current_mode = self.selected_mode
        print(current_mode)
        if current_mode == 'U-Net':
            self.modelName = "UNet_With_Original_DataSet"
            self.root_path = 'DataSets/OriginalDataSet/'
            self.IMAGE_CHANNELS = 38

        elif current_mode =='U-Net-RGB':
            self.modelName = "UNet_With_Naive_RGB_DataSet_Reduction"
            self.root_path = 'DataSets/RGBDataSet/'
            self.IMAGE_CHANNELS = 3     

        elif current_mode =='U-Net-PCA':
            self.modelName = "UNet_With_PCA_DataSet_Reduction"
            self.root_path = 'DataSets/PCADataSet/'
            self.IMAGE_CHANNELS = 3        

        elif current_mode =='U-Net-MNF':
            self.modelName = "UNet_With_MNF_DataSet_Reduction"
            self.root_path = 'DataSets/MNFDataSet/'
            self.IMAGE_CHANNELS = 3     

        elif current_mode =='U-Net-ICA':
            self.modelName = "UNet_With_ICA_DataSet_Reduction"
            self.root_path = 'DataSets/ICADataSet/'
            self.IMAGE_CHANNELS = 3    

        elif current_mode =='Swin-Unet':
            self.model_gui = "swinUnet"
            self.img_dir_gui = 'DataSets/OriginalDataSet/Set_1_images'
            self.out_dir = 'SegmentationResults/Swin-Unet'    

        elif current_mode =='UNet3+':
            self.model_gui = "unet3plus"
            self.img_dir_gui = 'DataSets/OriginalDataSet/Set_1_images'
            self.out_dir = 'SegmentationResults/UNet3+'    



    def runNet(self):

        # run the selected segmentation methods after choosing from method list
        if self.selected_mode == "Swin-Unet":
            gui_generate_img(self.model_gui ,self.img_dir_gui, self.out_dir)

        elif self.selected_mode == "UNet3+":
            gui_generate_img(self.model_gui ,self.img_dir_gui, self.out_dir)

        else:
            segment(self.modelName, self.root_path, self.IMAGE_CHANNELS)

    def clicker(self):

        # open the image
        fname = QFileDialog.getOpenFileName(self, "Open File", "DataSets", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg);;TIF Files (*.tif)")
        self.pixmap = QPixmap(fname[0])

        # QImage and QPixmap for zooming function 
        self.qimage = self.pixmap.toImage()  
        self.qpixmap = QPixmap(self.rgb_img.size()) 

        # convert pixmap to PIL image
        img = self.pixmap.toImage()
        buffer1 = QBuffer()
        buffer1.open(QBuffer.ReadWrite)
        img.save(buffer1, "PNG")
        self.Im0 = Image.open(io.BytesIO(buffer1.data()))

        # compute RGBA format image for blending purpose
        self.rgba_img = self.Im0.convert("RGBA")
        self.rgba_img = np.array(self.rgba_img)
        self.rgba_img = self.rgba_img.astype(float)
        print(f'opened image shape:',self.rgba_img.shape)
      
        # add image to label rgb_img
        self.rgb_img.setPixmap(self.pixmap)

    def rotation(self):

        # rotate the opened image 
        rotated_img = self.pixmap.transformed(QTransform().rotate(self.rotated_angle + 90))

        # add rotated image to label rgb_img
        self.rgb_img.setPixmap(rotated_img)

        # update the angle
        self.update_angle()
      
    def rotation2(self):

        # rotate the mask image 
        rotated_img2 = self.pixmap_mask.transformed(QTransform().rotate(self.rotated_angle + 90))

        # add rotated image to label modified_img
        self.modified_img.setPixmap(rotated_img2)

        # update the angle
        self.update_angle()
  
    def update_angle(self):

        # update angle for each rotation 
        self.rotated_angle = self.rotated_angle + 90
      
    def up_scaling(self):
        
        # scale the opened image
        self.pixmap1 = self.pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
    
        # add scaled image to label rgb_img
        self.rgb_img.setPixmap(self.pixmap1)

        # continue up scale image
        self.update_up_scaling()

    def down_scaling(self):
        
        # scale the opened image
        self.pixmap1 = self.pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
    
        # add scaled image to label rgb_img
        self.rgb_img.setPixmap(self.pixmap1)

        # continue down scale image
        self.update_down_scaling()

    def up_scaling2(self):
        
        # scale the mask image
        self.scaled2 = self.pixmap_mask.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
    
        # add scaled image to label modified_img
        self.modified_img.setPixmap(self.scaled2)
     
        # continue down scale image
        self.update_up_scaling()

    def down_scaling2(self):
        
        # scale the mask image
        self.scaled2 = self.pixmap_mask.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
    
        # add scaled image to label modified_img
        self.modified_img.setPixmap(self.scaled2)
    
        # continue down scale image
        self.update_down_scaling()

    def update_up_scaling(self):

        # update up-scaling value for each scaling 
        self.width = self.width + 20
        self.height = self.height + 20

    def update_down_scaling(self):

        # update down-scaling value for each scaling 
        self.width = self.width - 20
        self.height = self.height - 20
  
    def load_mask(self):

        # open image file from folder 
        fname = QFileDialog.getOpenFileName(self, "Open File", "SegmentationResults", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg);;TIF Files (*.tif)")
        print(fname[0])
     
        # import pathlib
        self.path = pathlib.Path(fname[0])
        self.path = self.path.parent
        self.path = self.path.__str__()
        print(self.path)
        

        # compute pixmap
        self.pixmap_mask = QPixmap(fname[0])
        # convert pixmap to QImage
        img = self.pixmap_mask.toImage()
        # create buffer
        buffer = QBuffer()
        # write buffer
        buffer.open(QBuffer.ReadWrite)
        # save image in buffer
        img.save(buffer, "PNG")
        # open saved image from buffer in PIL format
        self.mask0 = Image.open(io.BytesIO(buffer.data()))

        # compute RGBA format image for blending purpose
        self.rgba_mask = self.mask0.convert("RGBA")
        self.rgba_mask = np.array(self.rgba_mask)
        self.rgba_mask = self.rgba_mask.astype(float)
        print(f'mask shape:',self.rgba_mask.shape)
       
        # add image to label modified_img
        self.modified_img.setPixmap(self.pixmap_mask)

    def update_blending_mode(self):
        
        # select current blend mode
        self.selected_mode = self.comboMode.currentText()
        # print(f'current blend mode:',self.selected_mode)
      
    def blending(self):

        # get slider value
        self.value = self.slider.value()

        # set slider value to text
        self.slider_label.setText(str(self.value/10))

        # convert opened image into same mode of mask image
        Im1 = self.Im0.convert(self.mask0.mode)

        # convert opened image into same size of mask image
        Im1 = Im1.resize(self.mask0.size)

        # set current blend mode
        current_mode = self.selected_mode

        if current_mode == 'Overlay':
            
            # blend image
            blended_image = blend_modes.overlay(self.rgba_img, self.rgba_mask, self.value/10)

        elif current_mode =='Multiply':

            # blend image
            blended_image = blend_modes.multiply(self.rgba_img, self.rgba_mask, self.value/10)


        elif current_mode =='Lighten':
            
            # blend image
            blended_image = blend_modes.lighten_only(self.rgba_img, self.rgba_mask, self.value/10)

        else:
            blended_image = Image.blend(self.mask0,Im1,(self.value/10))

        # convert numpy array image to PIL image
        blended_image = np.uint8(blended_image)
        blended_image = Image.fromarray(blended_image)

        # convert PIL image to pixmap
        im = blended_image.convert("RGB")
        data = im.tobytes("raw","RGB")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)
        self.pixmap_out = QtGui.QPixmap.fromImage(qim)
      
        # show blended image
        self.modified_img.setPixmap(self.pixmap_out)

    def __connectEvents(self):

        # Mouse events
        self.rgb_img.mousePressEvent = self.mousePressAction
        self.rgb_img.mouseMoveEvent = self.mouseMoveAction
        self.rgb_img.mouseReleaseEvent = self.mouseReleaseAction

    def onResize(self):

        # resize the self.rgb_img 
        self.qpixmap = QPixmap(self.rgb_img.size())
        self.qpixmap.fill(QtCore.Qt.gray)
        self.qimage_scaled = self.qimage.scaled(self.rgb_img.width() * self.zoomX, self.rgb_img.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def update(self):
        
        #  draw the scaled image to self.rgb_img.
        if not self.qimage_scaled.isNull():
            # check if position is within limits to prevent unbounded panning.
            px, py = self.position
            px = px if (px <= self.qimage_scaled.width() - self.rgb_img.width()) else (self.qimage_scaled.width() - self.rgb_img.width())
            py = py if (py <= self.qimage_scaled.height() - self.rgb_img.height()) else (self.qimage_scaled.height() - self.rgb_img.height())
            px = px if (px >= 0) else 0
            py = py if (py >= 0) else 0
            self.position = (px, py)

            if self.zoomX == 1:
                self.qpixmap.fill(QtCore.Qt.white)

            # the act of painting the qpixamp
            painter = QPainter()
            painter.begin(self.qpixmap)
            painter.drawImage(QtCore.QPoint(0, 0), self.qimage_scaled,
                    QtCore.QRect(self.position[0], self.position[1], self.rgb_img.width(), self.rgb_img.height()) )
            painter.end()

            self.rgb_img.setPixmap(self.qpixmap)
        else:
            pass

    def mousePressAction(self, QMouseEvent):

        # track mouse clicking 
        x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
        if self.panFlag:
            self.pressed = QMouseEvent.pos()    # starting point of drag vector
            self.anchor = self.position         # save the pan position when panning starts

    def mouseMoveAction(self, QMouseEvent):

        # track mouse movement
        x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
        if self.pressed:
            dx, dy = x - self.pressed.x(), y - self.pressed.y()         # calculate the drag vector
            self.position = self.anchor[0] - dx, self.anchor[1] - dy    # update pan position using drag vector
            self.update()      
                                    
    def mouseReleaseAction(self, QMouseEvent):

        # release mouse action
        self.pressed = None            

    def zoomPlus(self):

        # zoom in 
        self.zoomX += 1
        px, py = self.position
        px += self.rgb_img.width()/2
        py += self.rgb_img.height()/2
        self.position = (px, py)
        self.qimage_scaled = self.qimage.scaled(self.rgb_img.width() * self.zoomX, self.rgb_img.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def zoomMinus(self):

        # zoom out
        if self.zoomX > 1:
            self.zoomX -= 1
            px, py = self.position
            px -= self.rgb_img.width()/2
            py -= self.rgb_img.height()/2
            self.position = (px, py)
            self.qimage_scaled = self.qimage.scaled(self.rgb_img.width() * self.zoomX, self.rgb_img.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()

    def resetZoom(self):

        # reset the zoom 
        self.zoomX = 1
        self.position = [0, 0]
        self.qimage_scaled = self.qimage.scaled(self.rgb_img.width() * self.zoomX, self.rgb_img.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def outSelect(self, Item):

        # show the selected item from table
        Item = self.table.currentRow()
        headItem = self.table.verticalHeaderItem(Item)
        currentItem = headItem.text()
        print(currentItem)

        # check if desired mask exists or not 
        if currentItem == 'Background':
            currentPath = self.path + '/0.png'

        elif currentItem == 'Blue dye':
            currentPath = self.path + '/1.png'

        elif currentItem == 'ICG':
            currentPath = self.path + '/2.png'

        elif currentItem == 'Specular reflection':
            currentPath = self.path + '/3.png'

        elif currentItem == 'Artery':
            currentPath = self.path + '/4.png'

        elif currentItem == 'Vein':
            currentPath = self.path + '/5.png'

        elif currentItem == 'Stroma':
            currentPath = self.path + '/6.png'

        elif currentItem == 'Artery,ICG':
            currentPath = self.path + '/7.png'

        elif currentItem == 'Stroma,ICG':
            currentPath = self.path + '/8.png'

        elif currentItem == 'Suture':
            currentPath = self.path + '/9.png'

        elif currentItem == 'Umbilical cord':
            currentPath = self.path + '/10.png'

        elif currentItem == 'Red dye':
            currentPath = self.path + '/11.png'
        
        try:
            currentMask = Image.open(currentPath)
            # show the selected mask, adjust desired figure size for convenience
            plt.figure(currentItem, figsize=(5,5))
            plt.axis('off')
            plt.imshow(currentMask)
            plt.draw()
            plt.pause(0.001)
        #if currentMask.exists():
        #if file.exists ():
            print ("File exist")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setText("Success!")
            msg.setInformativeText('Selected mask loaded')
            msg.setWindowTitle("Success!")
            msg.setBaseSize(QSize(200,145))
            msg.exec_()
        # compute RGBA format image for blending purpose
            self.rgba_mask = currentMask.convert("RGBA")
            self.rgba_mask = np.array(self.rgba_mask)
            self.rgba_mask = self.rgba_mask.astype(float)
        except:
            print ("File not exist")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setText("Fail!")
            msg.setInformativeText('Class does not exist')
            msg.setWindowTitle("Fail!")
            msg.setBaseSize(QSize(200,145))
            msg.exec_()


########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ########################################################################
    ##
    ########################################################################
    window = MainWindow()

    file = open("UI/idp.qss",'r')
    #### load style sheet####
    with file:
	    qss = file.read()
	    app.setStyleSheet(qss)
    window.show()
    sys.exit(app.exec_())
########################################################################
## END===>
########################################################################
