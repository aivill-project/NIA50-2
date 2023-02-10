class BoxController {
    constructor(mainController, boxModel) {
        this.mainController = mainController;
        this.boxModel = boxModel;
        this.boxListController = new BoxListController(mainController, this);
        this.boxEventController = new BoxEventController(mainController, this);
        
        /**
         * createBox, loadBox구분에 사용
         */
        this.isLoad = null;
    }

    /**
     * @description 전체 박스 생성 부분
     */
    async createBox() {
        this.boxModel.box3.set(
            this.boxModel.minVector,
            this.boxModel.maxVector
        );
        this.mainController.sceneModel.scene.add(this.boxModel.box3Helper);
        this.boxModel.box3Helper.layers.enable(1);
        await this.setBoxPoints(
            this.boxModel.minVector,
            this.boxModel.maxVector
        );

        // this.mainController.frameModel.boxIdIndex = 1;
        if (this.boxModel.subId !== -1) {
        }

        let selVal = this.boxListController.ClassArrayManager.getSelValue(
            this.mainController.clipOrder
        );
        if (selVal == "value1") {
        } else {
            this.boxModel.objType = "value2";
            this.boxModel.category = "MEDIAN_STRIP";
            this.boxModel.AtypicalYn = "y";
        }

        this.boxModel.boxOrder = `${this.mainController.mainModel.boxIdIndex}_-1`;
        this.boxModel.id = Number(this.mainController.mainModel.boxIdIndex);
        // this.boxModel.subId = this.mainController.frameModel.boxIdIndex;
        this.mainController.mainModel.boxIdIndex =
            Number(this.mainController.mainModel.boxIdIndex) + 1;

        await this.addArrowHelper();
        await this.addText();
        // await this.updateTextUsingTrackingId();
        this.boxListController.addList();
        await this.mainController.sceneController.setSubCameras(this);
        await this.select();
    }

    /**
     * @param {boolean?} isCopyed
     * JSON으로 불러온 데이터로 박스를 그릴때 사용하는 함수
     */
    async loadBox(isCopyed) {
        console.log("loadBox!");

        this.boxModel.box3.set(
            this.boxModel.minVector,
            this.boxModel.maxVector
        );
        // tw : 이거 주석 풀면 같은 Point Cloud 에 들어가게 됩니다...
        // this.boxModel.box3Helper.layers.set(1);
        this.mainController.sceneModel.scene.add(this.boxModel.box3Helper);
        this.boxModel.box3Helper.layers.enable(1);
        this.isLoad = isCopyed? false: true;
        await this.setBoxPoints(
            this.boxModel.minVector,
            this.boxModel.maxVector
        );

        // this.mainController.frameModel.boxIdIndex = 1;
        if (this.boxModel.subId !== -1) {
            this.boxModel.boxOrder = `${this.boxModel.id}_${this.boxModel.subId}`;
        } else {
            this.boxModel.boxOrder = `${this.boxModel.id}_-1`;
        }
        // this.mainController.mainModel.boxIdIndex = this.boxModel.id + 1;
        this.boxModel.box3Helper.rotation.y = this.boxModel.angle;
        await this.addArrowHelper();
        await this.addText();
        // await this.updateTextUsingTrackingId();
        this.boxListController.addList();
        // await this.select();
        // await this.mainController.sceneController.setSubCameras(this);
    }

    async setBoxPoints(minV, maxV) {
        let miV = minV.clone();
        let maV = maxV.clone();
        this.boxModel.boxPoints = {
            top: {
                leftFront: new THREE.Vector3(maV.x, maV.y, miV.z),
                leftRear: new THREE.Vector3(miV.x, maV.y, miV.z),
                rightFront: maV,
                rightRear: new THREE.Vector3(miV.x, maV.y, maV.z),
            },
            bottom: {
                leftFront: new THREE.Vector3(maV.x, miV.y, miV.z),
                leftRear: miV,
                rightFront: new THREE.Vector3(maV.x, miV.y, maV.z),
                rightRear: new THREE.Vector3(miV.x, miV.y, maV.z),
            },
            arrow: {
                front: new THREE.Vector3(
                    maV.x,
                    miV.y,
                    (maV.z - miV.z) / 2 + miV.z
                ),
                back: new THREE.Vector3(
                    (maV.x - miV.x) / 2 + miV.x,
                    miV.y,
                    (maV.z - miV.z) / 2 + miV.z
                ),
                arrowLeft: new THREE.Vector3(
                    maV.x - (maV.x - miV.x) / 8,
                    miV.y,
                    maV.z - (maV.z - miV.z) / 8
                ),
                arrowRight: new THREE.Vector3(
                    maV.x - (maV.x - miV.x) / 8,
                    miV.y,
                    (maV.z - miV.z) / 8 + miV.z
                ),
            },
        };

        await this.setGeometry(this.boxModel.boxPoints);
    }

    async setGeometry(boxPoints) {
        this.boxModel.geometry = [];

        this.boxModel.geometry.push(boxPoints.top.leftFront); //0
        this.boxModel.geometry.push(boxPoints.top.leftRear); //1
        this.boxModel.geometry.push(boxPoints.top.rightFront); // maxVector
        this.boxModel.geometry.push(boxPoints.top.rightRear); //
        this.boxModel.geometry.push(boxPoints.bottom.leftFront);
        this.boxModel.geometry.push(boxPoints.bottom.leftRear); //minVector
        this.boxModel.geometry.push(boxPoints.bottom.rightFront);
        this.boxModel.geometry.push(boxPoints.bottom.rightRear);

        this.boxModel.geometry.push(boxPoints.arrow.front);
        this.boxModel.geometry.push(boxPoints.arrow.back);
        this.boxModel.geometry.push(boxPoints.arrow.arrowLeft);
        this.boxModel.geometry.push(boxPoints.arrow.arrowRight);

        for (let i = 0; i < this.boxModel.geometry.length; i++) {
            this.rotate(this.boxModel.geometry[i], -this.boxModel.angle);
        }

        // this.boxModel.bufferGeometry = new THREE.BufferGeometry().setFromPoints(
        //     this.boxModel.geometry
        // );
        // console.log("this.geometry", this.boxModel.geometry);

        this.getBox2D();
        this.getDimension();
        this.getLocation();
        this.getDistance();
        this.getTrackingID();
        this.getYawOutput();

        //default는 lidar이기때문에 lidar2Radar가 수행되어야한다.
        let lMiV = Lidar2ImageManager.lidar2Radar(
            this.boxModel.minVector.x,
            this.boxModel.minVector.y,
            this.boxModel.minVector.z
        );

        let lMaV = Lidar2ImageManager.lidar2Radar(
            this.boxModel.maxVector.x,
            this.boxModel.maxVector.y,
            this.boxModel.maxVector.z
        );

        this.boxModel.radarMinVector = new THREE.Vector3(
            math.min(lMiV["x"], lMaV["x"]),
            math.min(lMiV["y"], lMaV["y"]),
            math.min(lMiV["z"], lMaV["z"])
        );

        this.boxModel.radarMaxVector = new THREE.Vector3(
            math.max(lMiV["x"], lMaV["x"]),
            math.max(lMiV["y"], lMaV["y"]),
            math.max(lMiV["z"], lMaV["z"])
        );

        // this.boxModel.radarMinVector = new THREE.Vector3(
        //   lMiV["x"],
        //   lMiV["y"],
        //   lMiV["z"]
        // );

        // this.boxModel.radarMaxVector = new THREE.Vector3(
        //   lMaV["x"],
        //   lMaV["y"],
        //   lMaV["z"]
        // );

        if (this.boxModel.box3Radar == null) {
            this.boxModel.box3Radar = new THREE.Box3(
                this.boxModel.radarMinVector,
                this.boxModel.radarMaxVector
            );
        } else {
            this.boxModel.box3Radar.set(
                this.boxModel.radarMinVector,
                this.boxModel.radarMaxVector
            );
        }
        if (this.boxModel.box3HelperRadar == null) {
            this.boxModel.box3HelperRadar = new THREE.Box3Helper(
                this.boxModel.box3Radar,
                0xff0000
            );
            this.boxModel.box3HelperRadar.rotation.y = this.boxModel.angle;
            //   this.boxModel.box3HelperRadar.layers.set(1);
            //   this.mainController.sceneModel.scene.add(this.boxModel.box3HelperRadar);
            //   this.arrowHelperRadar();
        } else {
            this.boxModel.box3HelperRadar.rotation.y = this.boxModel.angle;
        }

        this.mainController.imageController.drawCubicOnImage("NS");
        this.mainController.imageController.drawCubicOnImage();
        this.boxListController.insertBoxDataToTable();
    }

    async select() {
        this.mainController.imageController.initCanvasOnly2D();

        for (let i = 0; i < this.mainController.frameModel.boxes.length; i++) {
            const box = this.mainController.frameModel.boxes[i];
            if (box == this) {
                // this.mainController.sceneModel.transformControls.attach(
                //   this.mainController.frameModel.boxes[i].boxModel.box3Helper
                // );
                this.mainController.frameModel.setSelectBox(i);
                let selectedBox =
                    this.mainController.frameModel.getSelectedBox();
                this.mainController.imageController.reDrawObject();
                // tw : 박스 마우스 선택 시 drawCubic Image 설정
                this.mainController.imageController.drawCubicOnImage("NS");
                this.mainController.imageController.drawCubicOnImage();
                this.mainController.imageController.drawLabel2DAllNotSelect();
                this.mainController.imageController.reDrawPoints();
                await this.mainController.pointCloudController.checkPointsInSelectedBox();
                await this.mainController.pointCloudController.checkPointsInAllBox();
                await this.mainController.pointCloudController.checkPointsInSelectedBox_Lidar();
                await this.mainController.pointCloudController.checkPointsInAllBox_Lidar();
                await this.mainController.sceneController.setSubCameras(
                    selectedBox
                );
                this.boxModel.box3Helper.material.color.setHex(0x00fdfd);
                this.boxModel.box3HelperRadar.material.color.setHex(0x00fdfd);
                this.boxModel.box3Helper.layers.enable(1);

                document.querySelector(
                    `#obj_check_${this.boxModel.boxOrder}`
                ).checked = true;

                if (this.boxModel.area <= 900) {
                    document.querySelector(
                        `#obj_name_${this.boxModel.boxOrder}`
                    ).style.backgroundColor = MainModel.under900Color;
                } else {
                    document.querySelector(
                        `#obj_name_${this.boxModel.boxOrder}`
                    ).style.backgroundColor = "#3253C3";
                }

                document
                    .querySelector(`#obj_name_${this.boxModel.boxOrder}`)
                    .parentNode.classList.add("active");
            } else {
                if (box.boxModel.subId >= 0) {
                    //그룹이 있는경우
                    box.boxModel.box3Helper.material.color.setHex(0xffff00);
                    box.boxModel.box3HelperRadar.material.color.setHex(
                        0xffff00
                    );
                    box.boxModel.box3Helper.layers.disable(1);
                } else {
                    box.boxModel.box3Helper.material.color.setHex(0xff0000);
                    box.boxModel.box3HelperRadar.material.color.setHex(
                        0xff0000
                    );
                    box.boxModel.box3Helper.layers.disable(1);
                }
                document.querySelector(
                    `#obj_check_${box.boxModel.boxOrder}`
                ).checked = false;
                if (box.boxModel.area <= 900) {
                    document.querySelector(
                        `#obj_name_${box.boxModel.boxOrder}`
                    ).style.backgroundColor = MainModel.under900Color;
                } else {
                    document.querySelector(
                        `#obj_name_${box.boxModel.boxOrder}`
                    ).style.backgroundColor = "var(--main_bg02)";
                }

                document
                    .querySelector(`#obj_name_${box.boxModel.boxOrder}`)
                    .parentNode.classList.remove("active");
            }
        }

        document
            .querySelector(`#info_box_${this.boxModel.boxOrder}`)
            .scrollIntoViewIfNeeded(false);
        // this.mainController.frameController.frameModel.selectedBoxIndex = i;
    }

    async copy() {
        this.mainController.jsonBuffer.copyedBox = this;
    }

    async addArrowHelper() {
        //앞을 알려주는 빨간 화살표
        console.log("addArrowHelper");
        const dir = new THREE.Vector3(
            this.boxModel.dimension.length / 2 + 1,
            this.boxModel.dimension.height / 2,
            0
        );

        const origin = new THREE.Vector3(
            0,
            this.boxModel.dimension.height / 2,
            0
        );

        const direction = new THREE.Vector3().subVectors(dir, origin);
        const hex = "rgb(255,0,0)";

        this.boxModel.arrowHelper = new THREE.ArrowHelper(
            direction.normalize(),
            origin,
            direction.length(),
            hex
        );
        this.boxModel.arrowHelper.setLength(direction.length(), 0.3, 0.3);

        this.boxModel.box3Helper.add(this.boxModel.arrowHelper);
    }

    arrowHelperRadar() {
        return;
        const dirRadar = new THREE.Vector3(
            0,
            this.boxModel.dimension.height / 2,
            -math.abs(
                (this.boxModel.radarMinVector.z +
                    this.boxModel.radarMaxVector.z) /
                    2
            )
        );

        // let dirRadar = new THREE.Vector3();
        // this.boxModel.box3Radar.getWorldDirection(dirRadar);
        // dirRadar.y = this.boxModel.dimension.height / 2;

        const originRadar = new THREE.Vector3(
            0,
            this.boxModel.dimension.height / 2,
            0
        );

        const directionRadar = new THREE.Vector3().subVectors(
            dirRadar,
            originRadar
        );
        const hexRadar = "rgb(255,0,0)";

        this.boxModel.arrowHelperRadar = new THREE.ArrowHelper(
            directionRadar.normalize(),
            originRadar,
            directionRadar.length(),
            hexRadar
        );
        this.boxModel.arrowHelperRadar.setLength(
            directionRadar.length(),
            0.3,
            0.3
        );

        // if(index == 1){
        //   box.arrowHelper.layers.set(1);
        // }
        // this.boxModel.arrowHelperRadar.layers.set(1);

        // this.boxModel.arrowHelperRadar.traverse(function (node) {
        //   node.layers.set(1);
        // });
        // this.boxModel.box3HelperRadar.add(this.boxModel.arrowHelperRadar);
    }

    async addText() {
        // 박스의 텍스트 생성
        if (this.boxModel.text3D !== null) {
            this.mainController.sceneModel.scene.remove(this.boxModel.text3D);
            this.boxModel.text3D.geometry.dispose();
            this.boxModel.text3D.material.dispose();
        }
        if (this.mainController.frameModel.boxes.length == 0) {
            return;
        }

        let textContents = `${this.boxModel.id}`;
        const shapes = this.mainController.mainModel.fontjson.generateShapes(
            textContents,
            1
        );
        const geometry = new THREE.ShapeGeometry(shapes);

        geometry.computeBoundingBox();

        // let geometry = new THREE.TextGeometry(tracking_id,{font:fontjson.data.glyphs,size:2,height:0.00001});
        // let material = new THREE.MeshBasicMaterial({color:0x035b59});
        let material = new THREE.MeshBasicMaterial({ color: "red" });

        let text = new THREE.Mesh(geometry, material);
        // let boxcenter = await calc_boxcenter(box);
        text.rotation.x = -(Math.PI / 2); //빨간색은 X축
        text.rotation.z = -(Math.PI / 2); //파란색은 z축
        // // text.rotation.y = 90;
        // // text.rotation.z = 90;
        // // text.rotateZ = 90;
        // let boxoutput = box.output();
        let minV = this.boxModel.minVector.clone();
        let maxV = this.boxModel.maxVector.clone();
        let boxPosition = new THREE.Vector3(
            (minV.x + maxV.x) / 2,
            (minV.y + maxV.y) / 2,
            (minV.z + maxV.z) / 2
        );
        text.position.x = boxPosition.x + this.boxModel.dimension.length / 2;
        text.position.y = boxPosition.y + this.boxModel.dimension.height / 2;
        text.position.z = boxPosition.z + this.boxModel.dimension.width / 2;
        this.boxModel.text3D = text;
        this.mainController.sceneModel.scene.add(this.boxModel.text3D);
        // this.addTextArea();
    }

    async addTextArea(index) {
        // 박스 넓이 텍스트
        if (this.boxModel.textArea) {
            this.boxModel.text3D.remove(this.boxModel.textArea);
            this.boxModel.textArea.geometry.dispose();
            this.boxModel.textArea.material.dispose();
        }

        const dimension = this.boxModel.dimension;
        console.log("dimension", dimension);
        let area =
            Math.floor(
                dimension.width * dimension.height * dimension.length * 100
            ) / 100;
        let textContents = `${area}`;
        const shapes = this.mainController.mainModel.fontjson.generateShapes(
            textContents,
            1
        );
        const geometry = new THREE.ShapeGeometry(shapes);

        geometry.computeBoundingBox();

        let material = new THREE.MeshBasicMaterial({ color: "red" });

        let text = new THREE.Mesh(geometry, material);

        text.rotation.x = Math.PI / 2; //빨간색은 X축
        // text.rotation.z = -(Math.PI / 2); //파란색은 z축
        // text.rotation.y = -(Math.PI / 2); //파란색은 z축
        text.position.z = -1.5;

        // // text.rotation.y = 90;
        // // text.rotation.z = 90;
        // // text.rotateZ = 90;
        // let boxoutput = box.output();

        this.boxModel.text3D.add(text);
        this.boxModel.textArea = text;
        console.log("this.boxModel.textArea", this.boxModel.textArea);
    }

    async updateTextUsingTrackingId() {
        // 박스의 텍스트 최신화
        if (this.boxModel.text3D !== null) {
            this.mainController.sceneModel.scene.remove(this.boxModel.text3D);
            this.boxModel.text3D.geometry.dispose();
            this.boxModel.text3D.material.dispose();
            await this.addText();
        }

        return;
    }

    
    async getBox2D() {
        let box = this;
        console.log("getBox2D boxmodel",this.boxModel);
        this.boxModel.lineArray = [];
        this.boxModel.lineXArray = [];
        this.boxModel.lineYArray = [];


        for (var k = 0; k < box.boxModel.geometry.length; k++) {
            var point = Lidar2ImageManager.lidar2Image(
                box.boxModel.geometry[k].z,
                box.boxModel.geometry[k].x,
                box.boxModel.geometry[k].y
            );
            this.boxModel.lineArray.push(point);
            this.boxModel.lineXArray.push(point.x);
            this.boxModel.lineYArray.push(point.y);
            // if (point.x >= 0 && point.x <= 1920)
            //     this.boxModel.lineXArray.push(point.x);
            // if (point.y >= 0 && point.y <= 1200)
            //     this.boxModel.lineYArray.push(point.y);
        }

        let maxLineXArray = Math.max.apply(Math, this.boxModel.lineXArray) ;
        let minLineXArray = Math.min.apply(Math, this.boxModel.lineXArray);
        let maxLineYArray = Math.max.apply(Math, this.boxModel.lineYArray);
        let minLineYArray = Math.min.apply(Math, this.boxModel.lineYArray);
        
        if(this.isLoad){
            maxLineXArray = this.boxModel.box2D[0] + this.boxModel.box2D[2]/2;
            minLineXArray = this.boxModel.box2D[0] - this.boxModel.box2D[2]/2;
            maxLineYArray = this.boxModel.box2D[1] + this.boxModel.box2D[3]/2;
            minLineYArray = this.boxModel.box2D[1] - this.boxModel.box2D[3]/2;
        }
        this.boxModel.area =
            (maxLineXArray - minLineXArray) * (maxLineYArray - minLineYArray);
        
        this.isLoad = false;
        let selectedBox = this.mainController.frameModel.getSelectedBox();
        
        if (document.querySelector(`#obj_name_${this.boxModel.boxOrder}`)) {
            if (this.boxModel.area <= 900) {
                document.querySelector(
                    `#obj_name_${this.boxModel.boxOrder}`
                ).style.backgroundColor = MainModel.under900Color;
            } else if (this == selectedBox) {
                document.querySelector(
                    `#obj_name_${this.boxModel.boxOrder}`
                ).style.backgroundColor = "#3253C3";
            } else {
                document.querySelector(
                    `#obj_name_${this.boxModel.boxOrder}`
                ).style.backgroundColor = "var(--main_bg02)";
            }
        }
        console.log(
            "2D 이미지 사이즈 측정",
            minLineXArray,
            maxLineXArray,
            minLineYArray,
            maxLineYArray
        );

        if (maxLineXArray >= 1920) {
            maxLineXArray = 1920;
        }

        if (maxLineXArray < 0) {
            maxLineXArray = 0;
        }

        if (minLineXArray >= 1920) {
            minLineXArray = 1920;
        }

        if (minLineXArray < 0) {
            minLineXArray = 0;
        }

        if (maxLineYArray >= 1200) {
            maxLineYArray = 1200;
        }

        if (maxLineYArray < 0) {
            maxLineYArray = 0;
        }

        if (minLineYArray >= 1200) {
            minLineYArray = 1200;
        }

        if (minLineYArray < 0) {
            minLineYArray = 0;
        }

        // if (
        //     minLineXArray >= 1920 ||
        //     minLineYArray >= 1200 ||
        //     maxLineXArray < 0 ||
        //     maxLineYArray < 0
        // ) {
        //     minLineXArray = 0;
        //     maxLineXArray = 0;
        //     minLineYArray = 0;
        //     maxLineYArray = 0;
        // }

        this.boxModel.box2D = {
            xmin: minLineXArray,
            xmax: maxLineXArray,
            ymin: minLineYArray,
            ymax: maxLineYArray,
        };

        //json에 들어가는 값
        this.boxModel.box2DForJSON = [
            (minLineXArray + maxLineXArray) / 2,
            (minLineYArray + maxLineYArray) / 2,
            maxLineXArray - minLineXArray,
            maxLineYArray - minLineYArray,
        ];

        return;
    }

    async getDimension() {
        var v1 = this.boxModel.geometry[2].clone(); // max
        var v2 = this.boxModel.geometry[5].clone(); // min
        var v3 = this.boxModel.geometry[0].clone(); // topleft

        this.boxModel.dimension = {
            width: this.distance2D(v1, v3),
            height: v1.y - v2.y,
            length: this.distance2D(v2, v3),
        };

        return;
    }

    async getLocation() {
        var v1 = this.boxModel.geometry[2].clone(); // max
        var v2 = this.boxModel.geometry[5].clone(); // min
        const center = this.getCenter(v1, v2);
        // tw : getXcYcZc 는 다른 프로젝트의 calib 값 입니다.
        // 해당 프로젝트에서는 사용하지 않습니다.
        // const locationPoint = Lidar2ImageManager.getXcYcZc(
        //   center.x,
        //   -center.z,
        //   (v1.y + v2.y) / 2
        // );
        // 표출되어야 하는 내용
        //  x : center.x / y : -center.z / z : center.y

        const locationX = center.x;
        const locationY = (v1.y + v2.y) / 2;
        const locationZ = center.z;

        // const locationX = locationPoint[0][0];
        // const locationY = locationPoint[1][0];
        // const locationZ = locationPoint[2][0];
        this.boxModel.location = new THREE.Vector3(
            locationX,
            locationY,
            locationZ
        );
        // console.log("this.boxModel.location", this.boxModel.location);
        // this.boxModel.location = {
        //   x : locationX,
        //   y : locationY,
        //   z : locationZ
        // }

        return;
    }

    getTrackingID() {
        // this.boxModel.trackingId = box.tracking_id;
        // if (box.tracking_id == "") {
        //   this.tracking_id = '""';
        // }
        // this.subclass = box.subclass;
        // if (box.subclass == "none") {
        //   this.subclass = '""';
        // }
    }

    getDistance() {
        const v1 = this.boxModel.geometry[2].clone(); // max
        const v2 = this.boxModel.geometry[5].clone(); // min
        const center = this.getCenter(v1, v2);
        this.boxModel.distance = Math.sqrt(
            Math.pow(center.x, 2) + Math.pow(center.y, 2)
        );
    }

    getYawOutput() {
        let box = this;
        var point_1_for_yaw = Lidar2ImageManager.getXcYcZc(
            box.boxModel.geometry[2].z,
            box.boxModel.geometry[2].x,
            box.boxModel.geometry[2].y
        );
        var point_2_for_yaw = Lidar2ImageManager.getXcYcZc(
            box.boxModel.geometry[0].z,
            box.boxModel.geometry[0].x,
            box.boxModel.geometry[0].y
        );
        // var point_1_for_yaw = Lidar2ImageManager.getXcYcZc(box.geometry.vertices[0].x, -box.geometry.vertices[0].z, box.geometry.vertices[0].y);
        // var point_2_for_yaw = Lidar2ImageManager.getXcYcZc(box.geometry.vertices[2].x, -box.geometry.vertices[2].z, box.geometry.vertices[2].y);
        var target_vector_for_yaw = new THREE.Vector3(
            point_2_for_yaw[0][0] - point_1_for_yaw[0][0],
            point_2_for_yaw[1][0] - point_1_for_yaw[1][0],
            point_2_for_yaw[2][0] - point_1_for_yaw[2][0]
        );

        target_vector_for_yaw.y = 0;
        this.boxModel.yaw = this.getYaw(target_vector_for_yaw) * -1;
        return;
    }

    getYaw(target_vector_for_get_yaw) {
        var origin_vector_for_get_yaw = new THREE.Vector3(1, 0, 0);

        target_vector_for_get_yaw.y = 0;

        var yaw = target_vector_for_get_yaw.angleTo(origin_vector_for_get_yaw);

        if (target_vector_for_get_yaw.z < 0) {
            yaw = yaw * -1;
        }
        return yaw;
    }

    /**
     * get Center In BOX
     * @param {Vector} v1
     * @param {Vector} v2
     * @returns
     */

    distance2D(v1, v2) {
        return Math.pow(
            Math.pow(v1.x - v2.x, 2) + Math.pow(v1.z - v2.z, 2),
            0.5
        );
    }

    getCenter(v1, v2) {
        return new THREE.Vector3((v1.x + v2.x) / 2.0, 0.0, (v1.z + v2.z) / 2.0);
    }

    getMin(v1, v2) {
        return new THREE.Vector3(
            Math.min(v1.x, v2.x),
            Math.min(v1.y, v2.y),
            Math.min(v1.z, v2.z)
        );
    }

    getMax(v1, v2) {
        return new THREE.Vector3(
            Math.max(v1.x, v2.x),
            Math.max(v1.y, v2.y),
            Math.max(v1.z, v2.z)
        );
    }

    /**
     * Box Rotate
     * @param {Vector} v1
     * @param {Vector} v2
     * @param {number} angle
     */
    rotate(point, angle) {
        const center = this.getCenter(
            this.boxModel.minVector.clone(),
            this.boxModel.maxVector.clone()
        );
        let zeroX = point.x - center.x;
        let zeroZ = point.z - center.z;

        let rotateMatrix = [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ];
        let zeroPoint = [[zeroX], [zeroZ]];

        let rotatePoint = math.multiply(rotateMatrix, zeroPoint);

        let resultX = rotatePoint[0][0] + center.x;
        let resultZ = rotatePoint[1][0] + center.z;

        point.x = resultX;
        point.z = resultZ;
        // v1.sub(center);
        // v2.sub(center);
        // let temp1 = v1.clone();
        // let temp2 = v2.clone();

        // v1.x = Math.cos(angle) * temp1.x - Math.sin(angle) * temp1.z;
        // v2.x = Math.cos(angle) * temp2.x - Math.sin(angle) * temp2.z;

        // v1.z = Math.sin(angle) * temp1.x + Math.cos(angle) * temp1.z;
        // v2.z = Math.sin(angle) * temp2.x + Math.cos(angle) * temp2.z;

        // v1.add(center);
        // v2.add(center);
    }
}
