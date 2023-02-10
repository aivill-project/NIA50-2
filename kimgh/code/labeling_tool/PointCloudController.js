class PointCloudController {
    constructor(mainController) {
        this.mainController = mainController;
        this.sceneModel = mainController.sceneModel;
        this.greenMesh; //radar
        this.prevPointsLidar; //lidar
        this.greenPointSize = 3.5;
        this.pointCloudSize = 2.5;
        this.pointCloudSizeLidar = Number(
            document.getElementById("lidarRange").value
        );
        this.pointCloudSizeRadar = Number(
            document.getElementById("radarRange").value
        );
        this.addEvent();
    }

    async addEvent() {
        document
            .getElementById("radarRange")
            .addEventListener("change", (e) => {
                // document.getElementById("speedValueP").innerText = this.value;
                if (!this.sceneModel.pointcloud) {
                    return;
                }
                console.log("변환", this, e);
                this.sceneModel.pointcloud.material.size = Number(
                    e.target.value
                );
                this.pointCloudSizeRadar = Number(e.target.value);
                for (
                    let i = 0;
                    i < this.mainController.sceneModel.scene.children.length;
                    i++
                ) {
                    let object =
                        this.mainController.sceneModel.scene.children[i];
                    if (object.name == `greenMesh`) {
                        object.material.size = Number(e.target.value) + 0.5;
                    }
                }
            });

        document
            .getElementById("lidarRange")
            .addEventListener("change", (e) => {
                // document.getElementById("speedValueP").innerText = this.value;
                if (!this.sceneModel.pointcloudLidar) {
                    return;
                }
                console.log("변환", this, e);
                this.sceneModel.pointcloudLidar.material.size = Number(
                    e.target.value
                );
                this.pointCloudSizeLidar = Number(e.target.value);
                for (
                    let i = 0;
                    i < this.mainController.sceneModel.scene.children.length;
                    i++
                ) {
                    let object =
                        this.mainController.sceneModel.scene.children[i];
                    if (object.name == `greenMesh_Lidar`) {
                        object.material.size = Number(e.target.value) + 0.5;
                    }
                }
            });
        document.getElementById("radarRange").oninput = function () {
            document.getElementById("radarRangeP").innerText = this.value;
        };

        document.getElementById("lidarRange").oninput = function () {
            document.getElementById("lidarRangeP").innerText = this.value;
        };
        document
            .getElementById("sliderOkButton")
            .addEventListener("click", (e) => {
                document.getElementById("pointResizeModal").style.display =
                    "none";
            });

        document
            .getElementById("pointResizeToggle")
            .addEventListener("click", (e) => {
                if (
                    document.getElementById("pointResizeModal").style.display ==
                    "none"
                ) {
                    document.getElementById("pointResizeModal").style.display =
                        "block";
                } else {
                    document.getElementById("pointResizeModal").style.display =
                        "none";
                }
            });
    }

    async load_PCD_file(file) {
        //radar
        if (this.sceneModel.pointcloud !== null) {
            this.sceneModel.scene.remove(this.sceneModel.pointcloud);
            this.sceneModel.pointcloud.geometry.dispose();
            this.sceneModel.pointcloud.material.dispose();
            this.sceneModel.pointcloud = null;
        }
        const loader = new THREE.PCDLoader();
        // load a resource
        loader.load(
            // resource URL
            file,
            // called when the resource is loaded
            async (mesh) => {
                mesh.name = "pointcloudRadar";
                let pointColors = await this.setPointcloudColor(mesh, "yellow");
                // await this.sleep(1);
                this.sceneModel.pointcloud = await this.setMesh(
                    mesh,
                    pointColors
                );
                await this.setRadar2LidarPointCloud(mesh);
            },
            // called when loading is in progresses
            function (xhr) {
                console.log((xhr.loaded / xhr.total) * 100 + "% loaded");
            },
            // called when loading has errors
            function (error) {
                console.log("An error happened");
            }
        );
    }

    /**
     * @description 라이다 파일을 받아 three.js화면에 띄어주는 함수
     * @param {*} file 라이다 파일 하나를 받아옴
     */
    async load_PCD_file_Lidar(file) {
        if (this.sceneModel.pointcloudLidar !== null) {
            this.sceneModel.scene.remove(this.sceneModel.pointcloudLidar);
            this.sceneModel.pointcloudLidar.geometry.dispose();
            this.sceneModel.pointcloudLidar.material.dispose();
            this.sceneModel.pointcloudLidar = null;
        }
        const loader = new THREE.PCDLoader();
        // load a resource
        loader.load(
            // resource URL
            file,
            // called when the resource is loaded
            async (mesh) => {
                let realMesh = await this.setPointCloudGeometry(mesh);
                realMesh.name = "pointcloudLidar";
                let pointColors = await this.setPointcloudColor(
                    realMesh,
                    "white"
                );
                // await this.sleep(1);
                this.sceneModel.pointcloudLidar = await this.setMesh(
                    realMesh,
                    pointColors
                );
                await this.mainController.frameController.loadDataFromBuffer();
                MainModel.canFrameEdit = true;
                console.log("MainModel", MainModel.canFrameEdit);
                this.mainController.offCutton();
            },
            // called when loading is in progresses
            function (xhr) {
                console.log((xhr.loaded / xhr.total) * 100 + "% loaded");
            },
            // called when loading has errors
            function (error) {
                console.log("An error happened");
                this.mainController.offCutton();
            }
        );
    }

    /**
     *
     * @param {*} mesh
     */
    async setRadar2LidarPointCloud(mesh) {
        try {
            let positions = mesh.geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i += 3) {
                // let Point = pointcloud.geometry.vertices[i].clone();
                let newPoint = Lidar2ImageManager.lidar2RadarScene(
                    positions[i],
                    positions[i + 1],
                    positions[i + 2]
                );

                positions[i] = newPoint.x;
                positions[i + 1] = newPoint.y;
                positions[i + 2] = newPoint.z;
            }

            mesh.geometry.attributes.position.needsUpdate = true;
        } catch (e) {
            console.error(e);
        }
    }

    /**
     * @description selectbox와 똑같은 object를 만들기 위한용도
     * @param {*} box
     * @param {string} out string값을 넣기만하면 박스 밖부분도 가져옴
     * @returns
     */
    async makeCube(box, out) {
        let geometry;
        if (out) {
            geometry = new THREE.BoxGeometry(
                box.boxModel.maxVector.x + 3 - (box.boxModel.minVector.x - 3),
                box.boxModel.maxVector.y + 3 - (box.boxModel.minVector.y - 3),
                box.boxModel.maxVector.z + 3 - (box.boxModel.minVector.z - 3)
            );
        } else {
            geometry = new THREE.BoxGeometry(
                box.boxModel.maxVector.x - box.boxModel.minVector.x,
                box.boxModel.maxVector.y - box.boxModel.minVector.y,
                box.boxModel.maxVector.z - box.boxModel.minVector.z
            );
        }

        const material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
        });
        const cube = new THREE.Mesh(geometry, material);
        const center = new THREE.Vector3(
            (box.boxModel.minVector.x + box.boxModel.maxVector.x) / 2,
            (box.boxModel.minVector.y + box.boxModel.maxVector.y) / 2,
            (box.boxModel.minVector.z + box.boxModel.maxVector.z) / 2
        );
        cube.position.set(center.x, center.y, center.z);
        cube.rotation.y = box.boxModel.angle;
        cube.name = "cube";

        cube.updateMatrixWorld();

        return cube;
    }

    async setPointCloudGeometry(mesh) {
        let returnArray = [];
        for (
            let i = 0;
            i < mesh.geometry.attributes.position.array.length;
            i += 3
        ) {
            let x = mesh.geometry.attributes.position.array[i];
            let y = mesh.geometry.attributes.position.array[i + 1];
            let z = mesh.geometry.attributes.position.array[i + 2];
            if (
                y / x < -1.73205 ||
                y / x > 1.73205 ||
                x < 0 ||
                x * x + y * y > 6400
            ) {
                continue;
            }
            let v = new THREE.Vector3(x, y, z);
            returnArray.push(v.x, v.y, v.z);
        }
        mesh.geometry.attributes.position.array = returnArray;
        mesh.geometry.attributes.position.needsUpdate = true;
        mesh.geometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(returnArray, 3)
        );
        console.log("setPointCloudGeometry", mesh);
        return mesh;
    }

    /** color
     * 128 * 2 씩 로드
     */
    async setPointcloudColor(mesh, color) {
        let returnArray = [];
        for (
            let i = 0;
            i < mesh.geometry.attributes.position.array.length;
            i += 3
        ) {
            let x = mesh.geometry.attributes.position.array[i];
            let y = mesh.geometry.attributes.position.array[i + 1];
            let z = mesh.geometry.attributes.position.array[i + 2];
            let v = new THREE.Vector3(x, y, z);

            let t = await this.overDistanceRedColor(v, color, mesh.name);
            returnArray.push(t[0], t[1], t[2]);
        }
        return returnArray;
    }

    async setMesh(mesh, colorArray) {
        if (mesh.name.indexOf("Lidar") != -1) {
            mesh.material.size = this.pointCloudSizeLidar;
        } else {
            mesh.material.size = this.pointCloudSizeRadar;
        }
        mesh.material.vertexColors = THREE.VertexColors;
        mesh.material.sizeAttenuation = false;
        mesh.geometry.setAttribute(
            "color",
            new THREE.Float32BufferAttribute(colorArray, 3)
        );
        mesh.geometry.attributes.color.needsUpdate = true;
        // if (mesh.name.indexOf("Radar") !== -1) {
        //   mesh.layers.enable(1);
        // }
        this.sceneModel.scene.add(mesh);

        mesh.rotateX((-90 * Math.PI) / 180); // X축으로 돌림
        // mesh.rotateZ((-89.6 * Math.PI) / 180);
        return mesh;
    }

    async overDistanceRedColor(v, color, meshName) {
        //150m이상의 pointcloud는 빨간색 처리
        var origin_point = new THREE.Vector3(0, 0, 0);
        var target_point_for_distance_measurement = v.clone();
        var returnColor = [255, 255, 255];
        if (color == "white") {
            returnColor = [255, 255, 255];
        } else if (color == "black") {
            returnColor = [0, 0, 0];
        } else if (color == "yellow") {
            returnColor = [255, 255, 0];
        }

        if (meshName.indexOf("Radar") !== -1) {
            return returnColor;
        }

        target_point_for_distance_measurement.y = 0;
        var distance = origin_point.distanceTo(
            target_point_for_distance_measurement
        );

        if (distance >= 80) {
            // // console.log("붉");
            // return new THREE.Color('red');
            returnColor = [255, 0, 0];
            return returnColor;
        } else {
            return returnColor;
        }
    }

    //radar
    async checkPointsInSelectedBox() {
        try {
            //radar
            console.log("checkPointsInSelectedBox");
            //selectedBox와 pointcloud를 읽어서 selectedBox내의 pointcloud를 초록색으로 바꾸는 함수.

            for (
                let i = 0;
                i < this.mainController.sceneModel.scene.children.length;
                i++
            ) {
                let object = this.mainController.sceneModel.scene.children[i];
                if (object.name == `greenMesh`) {
                    this.mainController.sceneModel.scene.remove(object);
                    object.geometry.dispose();
                    object.material.dispose();
                }
            }

            // 생성된 박스가 없을때
            if (this.mainController.frameModel.boxes.length == 0) {
                return;
            }

            // // 선택된 박스가 없을때
            // if (selectedBox == null) {
            //   return;
            // }

            const selectedBox = this.mainController.frameModel.getSelectedBox();
            let cube = await this.makeCube(selectedBox);
            let boxMatrixInverse = new THREE.Matrix4().getInverse(
                cube.clone().matrixWorld
            );
            let inverseBox = cube.clone();

            inverseBox.applyMatrix4(boxMatrixInverse);

            let boxTemp = new THREE.Box3().setFromObject(inverseBox);

            // // console.log("point cloud", pointcloud);

            // for(let i =0; i< prev_points.length; i += 3){

            //     if(flag == 0 ){
            //         pointcloud.geometry.attributes.color.array[prev_points[i]] = 255;
            //         pointcloud.geometry.attributes.color.array[prev_points[i+1]] = 255;
            //         pointcloud.geometry.attributes.color.array[prev_points[i+2]] = 255;
            //     }
            // }
            var green_geometry_vertices = [];
            var green_geometry_colors = [];
            // prev_points = [];
            let positions =
                this.mainController.sceneModel.pointcloud.geometry.attributes
                    .position.array;
            let greenPointLength = 0;
            for (let i = 0; i < positions.length; i += 3) {
                // let Point = pointcloud.geometry.vertices[i].clone();
                let Point = new THREE.Vector3(
                    positions[i],
                    positions[i + 1],
                    positions[i + 2]
                );
                let axis = new THREE.Vector3(1, 0, 0);

                Point.applyAxisAngle(axis, (-90 * Math.PI) / 180);
                Point.applyMatrix4(boxMatrixInverse);

                if (boxTemp.containsPoint(Point)) {
                    // console.log("isInsideisInside!!", i);
                    // pointcloud.geometry.attributes.color.array[i] = 0;
                    // pointcloud.geometry.attributes.color.array[i+1] = 128;
                    // pointcloud.geometry.attributes.color.array[i+2] = 0;
                    green_geometry_vertices.push(Point);
                    greenPointLength++;
                    // prev_points.push(i,i+1,i+2);
                    green_geometry_colors.push(255, 255, 0);
                }
            }
            selectedBox.boxModel.greenPointsRadarLength = greenPointLength;
            console.log("RadarGreen 점군 갯수", greenPointLength);
            let bufferGeometry = new THREE.BufferGeometry().setFromPoints(
                green_geometry_vertices
            );
            bufferGeometry.setAttribute(
                "color",
                new THREE.Float32BufferAttribute(green_geometry_colors, 3)
            );

            var material = new THREE.PointsMaterial({
                size: this.pointCloudSizeRadar + 0.5,
                sizeAttenuation: false,
                vertexColors: THREE.VertexColors,
            });
            this.greenMesh = new THREE.Points(bufferGeometry, material);
            const center = new THREE.Vector3(
                (selectedBox.boxModel.minVector.x +
                    selectedBox.boxModel.maxVector.x) /
                    2,
                (selectedBox.boxModel.minVector.y +
                    selectedBox.boxModel.maxVector.y) /
                    2,
                (selectedBox.boxModel.minVector.z +
                    selectedBox.boxModel.maxVector.z) /
                    2
            );
            this.greenMesh.position.set(center.x, center.y, center.z);
            this.greenMesh.rotation.y = selectedBox.boxModel.angle;
            this.greenMesh.geometry.attributes.position.needsUpdate = true;
            this.greenMesh.name = "greenMesh";
            this.greenMesh.layers.enable(1);
            //   this.greenMesh.layers.set(1);
            this.mainController.sceneModel.scene.add(this.greenMesh);
            // this.mainController.sceneModel.pointcloud.geometry.attributes.color.needsUpdate = true;

            cube.geometry.dispose();
            cube.material.dispose();
        } catch (e) {
            console.error(e);
        }
    }

    //radar
    async checkPointsInAllBox() {
        return;
        try {
            //radar
            console.log("checkPointsInAllBox");
            //selectedBox와 pointcloud를 읽어서 selectedBox내의 pointcloud를 초록색으로 바꾸는 함수.
            for (
                let i = 0;
                i < this.mainController.sceneModel.scene.children.length;
                i++
            ) {
                let object = this.mainController.sceneModel.scene.children[i];
                if (object.name == `greenMesh_Not_Selected`) {
                    this.mainController.sceneModel.scene.remove(object);
                    object.geometry.dispose();
                    object.material.dispose();
                }
            }
            // 생성된 박스가 없을때
            if (this.mainController.frameModel.boxes.length == 0) {
                return;
            }

            // // 선택된 박스가 없을때
            // if (selectedBox == null) {
            //   return;
            // }

            // console.log(
            //   "gMNS",
            //   this.mainController.sceneModel.scene.getObjectByName(
            //     `greenMesh_Not_Selected`
            //   ),
            //   this.mainController.sceneModel.scene
            // );
            const selectedBox = this.mainController.frameModel.getSelectedBox();
            for (
                let k = 0;
                k < this.mainController.frameModel.boxes.length;
                k++
            ) {
                if (this.mainController.frameModel.boxes[k] == selectedBox) {
                    continue;
                } else {
                    let green_geometry_vertices = [];
                    let green_geometry_colors = [];
                    let box = this.mainController.frameModel.boxes[k];
                    const geometry = new THREE.BoxGeometry(
                        box.boxModel.dimension.width,
                        box.boxModel.dimension.height,
                        box.boxModel.dimension.length
                    );
                    const material = new THREE.MeshBasicMaterial({
                        color: 0x00ff00,
                    });
                    const cube = new THREE.Mesh(geometry, material);
                    const center = new THREE.Vector3(
                        (box.boxModel.radarMinVector.x +
                            box.boxModel.radarMaxVector.x) /
                            2,
                        (box.boxModel.radarMinVector.y +
                            box.boxModel.radarMaxVector.y) /
                            2,
                        (box.boxModel.radarMinVector.z +
                            box.boxModel.radarMaxVector.z) /
                            2
                    );
                    cube.position.set(center.x, center.y, center.z);
                    cube.rotation.y = box.boxModel.angle;
                    cube.name = "cube";

                    cube.updateMatrixWorld();

                    var boxMatrixInverse = new THREE.Matrix4().getInverse(
                        cube.clone().matrixWorld
                    );
                    var inverseBox = cube.clone();

                    inverseBox.applyMatrix4(boxMatrixInverse);

                    var boxTemp = new THREE.Box3().setFromObject(inverseBox);

                    // // console.log("point cloud", pointcloud);

                    // for(let i =0; i< prev_points.length; i += 3){

                    //     if(flag == 0 ){
                    //         pointcloud.geometry.attributes.color.array[prev_points[i]] = 255;
                    //         pointcloud.geometry.attributes.color.array[prev_points[i+1]] = 255;
                    //         pointcloud.geometry.attributes.color.array[prev_points[i+2]] = 255;
                    //     }
                    // }

                    // prev_points = [];
                    let positions =
                        this.mainController.sceneModel.pointcloud.geometry
                            .attributes.position.array;
                    for (let i = 0; i < positions.length; i += 3) {
                        // let Point = pointcloud.geometry.vertices[i].clone();
                        let Point = new THREE.Vector3(
                            positions[i],
                            positions[i + 1],
                            positions[i + 2]
                        );
                        let axis = new THREE.Vector3(1, 0, 0);
                        Point.applyAxisAngle(axis, (-90 * Math.PI) / 180);

                        Point.applyMatrix4(boxMatrixInverse);

                        if (boxTemp.containsPoint(Point)) {
                            // console.log("isInsideisInside!!", i);
                            // pointcloud.geometry.attributes.color.array[i] = 0;
                            // pointcloud.geometry.attributes.color.array[i+1] = 128;
                            // pointcloud.geometry.attributes.color.array[i+2] = 0;
                            green_geometry_vertices.push(Point);

                            // prev_points.push(i,i+1,i+2);
                            green_geometry_colors.push(0, 0.5, 0);
                        }
                    }

                    let bufferGeometry =
                        new THREE.BufferGeometry().setFromPoints(
                            green_geometry_vertices
                        );
                    bufferGeometry.setAttribute(
                        "color",
                        new THREE.Float32BufferAttribute(
                            green_geometry_colors,
                            3
                        )
                    );

                    var greenMeshMaterial = new THREE.PointsMaterial({
                        size: 2,
                        sizeAttenuation: false,
                        vertexColors: THREE.VertexColors,
                    });

                    let greenMeshNotSelected = new THREE.Points(
                        bufferGeometry,
                        greenMeshMaterial
                    );
                    greenMeshNotSelected.position.set(
                        center.x,
                        center.y,
                        center.z
                    );
                    greenMeshNotSelected.rotation.y = box.boxModel.angle;
                    greenMeshNotSelected.geometry.attributes.position.needsUpdate = true;
                    greenMeshNotSelected.name = `greenMesh_Not_Selected`;
                    //   greenMeshNotSelected.layers.set(1);
                    this.mainController.sceneModel.scene.add(
                        greenMeshNotSelected
                    );
                    // this.mainController.sceneModel.pointcloud.geometry.attributes.color.needsUpdate = true;

                    cube.geometry.dispose();
                    cube.material.dispose();
                }
            }
        } catch (e) {
            console.error(e);
        }
    }
    /**
     * @description 선택되지 않은 박스중 라이다 포인트가 박스내에있으면 포인트클라우드 색을 바꿈
     * @returns
     */
    async checkPointsInAllBox_Lidar() {
        //selectedBox와 pointcloud를 읽어서 selectedBox내의 pointcloud를 초록색으로 바꾸는 함수.
        try {
            console.log("checkPointsInAllBox_Lidar");
            //   await this.deleteGreenMesh(`greenMesh_Lidar_Not_Selected`);

            await this.makeGreenPoint("lidar");
            // 생성된 박스가 없을때
            if (this.mainController.frameModel.boxes.length == 0) {
                await this.makePointCloudWhite(
                    this.mainController.sceneModel.pointcloudLidar
                );
                return;
            }
        } catch (e) {
            console.error(e);
        }
    }

    /**
     * @description 박스내 라이다 포인트 들로 green컬러 mesh를 만드는함수
     * @returns
     */
    async checkPointsInSelectedBox_Lidar() {
        console.log("checkPointsInSelectedBox_Lidar");
        try {
            //selectedBox와 pointcloud를 읽어서 selectedBox내의 pointcloud를 초록색으로 바꾸는 함수.
            this.deleteGreenMesh(`greenMesh_Lidar`);

            // 생성된 박스가 없을때
            if (this.mainController.frameModel.boxes.length == 0) {
                return;
            }
            const selectedBox = this.mainController.frameModel.getSelectedBox();

            let cubeOut = await this.makeCube(selectedBox, "1");
            let boxMatrixInverseOut = new THREE.Matrix4().getInverse(
                cubeOut.clone().matrixWorld
            );
            let inverseBoxOut = cubeOut.clone();
            inverseBoxOut.applyMatrix4(boxMatrixInverseOut);

            let boxTempOut = new THREE.Box3().setFromObject(inverseBoxOut);

            let cube = await this.makeCube(selectedBox);
            let boxMatrixInverse = new THREE.Matrix4().getInverse(
                cube.clone().matrixWorld
            );
            let inverseBox = cube.clone();

            inverseBox.applyMatrix4(boxMatrixInverse);

            let boxTemp = new THREE.Box3().setFromObject(inverseBox);

            let greenGeometry = [];
            let greenColors = [];

            let outGeometry = [];
            let outColors = [];
            // prev_points = [];
            let positions =
                this.mainController.sceneModel.pointcloudLidar.geometry
                    .attributes.position.array;
            let greenPointLength = 0;
            for (let i = 0; i < positions.length; i += 3) {
                // let Point = pointcloud.geometry.vertices[i].clone();
                let Point = new THREE.Vector3(
                    positions[i],
                    positions[i + 1],
                    positions[i + 2]
                );
                let axis = new THREE.Vector3(1, 0, 0);

                Point.applyAxisAngle(axis, (-90 * Math.PI) / 180);
                Point.applyMatrix4(boxMatrixInverse);

                if (boxTemp.containsPoint(Point)) {
                    // console.log("isInsideisInside!!", i);
                    // pointcloud.geometry.attributes.color.array[i] = 0;
                    // pointcloud.geometry.attributes.color.array[i+1] = 128;
                    // pointcloud.geometry.attributes.color.array[i+2] = 0;
                    greenGeometry.push(Point);
                    greenPointLength++;
                    // prev_points.push(i,i+1,i+2);

                    // greenColors.push(0, 0.5, 0); //
                    greenColors.push(0, 2, 0); //
                } else if (boxTempOut.containsPoint(Point)) {
                    outGeometry.push(Point);
                    outColors.push(128, 128, 128); //
                }
            }
            selectedBox.boxModel.greenPointsLidarLength = greenPointLength;
            console.log("LidarGreen 점군 갯수", greenPointLength);
            this.makeGreenMesh(greenGeometry, greenColors, selectedBox);
            this.makeGreenMesh(
                outGeometry,
                outColors,
                selectedBox,
                "아무거나",
                2
            );

            cube.geometry.dispose();
            cube.material.dispose();
        } catch (e) {
            console.error(e);
        }
    }

    async deleteGreenMesh(name) {
        for (
            let i = this.mainController.sceneModel.scene.children.length - 1;
            i >= 0;
            i--
        ) {
            let object = this.mainController.sceneModel.scene.children[i];
            if (object.name == name) {
                this.mainController.sceneModel.scene.remove(object);
                object.geometry.dispose();
                object.material.dispose();
            }
        }
        return;
    }

    /**
     * @description 현재 lidar에서만 사용중이며 조건이 만족한 라이다 포인트클라우드 색을 green으로 바꿈
     * @param {*} name
     */
    async makeGreenPoint(name) {
        let pointcloud;
        let prevPoints;
        if (name.indexOf("lidar") != -1) {
            pointcloud = this.mainController.sceneModel.pointcloudLidar;
            prevPoints = this.prevPointsLidar;
        } else {
            pointcloud = this.mainController.sceneModel.pointcloudLidar;
            prevPoints = this.prevPointsLidar;
        }

        // this.deleteGreenMesh(`greenMesh_Lidar_Not_Selected`)

        // // greenpoint 를 다시 흰색 처리;
        if (prevPoints) {
            for (let i = 0; i < prevPoints.length; i += 3) {
                pointcloud.geometry.attributes.color.array[prevPoints[i]] = 255;
                pointcloud.geometry.attributes.color.array[
                    prevPoints[i + 1]
                ] = 255;
                pointcloud.geometry.attributes.color.array[
                    prevPoints[i + 2]
                ] = 255;
            }
        }
        this.prevPointsLidar = [];
        //
        const selectedBox = this.mainController.frameModel.getSelectedBox();
        for (let k = 0; k < this.mainController.frameModel.boxes.length; k++) {
            if (this.mainController.frameModel.boxes[k] == selectedBox) {
                continue;
            } else {
                let green_geometry_vertices = [];
                let green_geometry_colors = [];
                let box = this.mainController.frameModel.boxes[k];

                let cube = await this.makeCube(box);
                let boxMatrixInverse = new THREE.Matrix4().getInverse(
                    cube.clone().matrixWorld
                );
                let inverseBox = cube.clone();

                inverseBox.applyMatrix4(boxMatrixInverse);

                var boxTemp = new THREE.Box3().setFromObject(inverseBox);

                let positions = pointcloud.geometry.attributes.position.array;
                for (let i = 0; i < positions.length; i += 3) {
                    let Point = new THREE.Vector3(
                        positions[i],
                        positions[i + 1],
                        positions[i + 2]
                    );
                    let axis = new THREE.Vector3(1, 0, 0);

                    Point.applyAxisAngle(axis, (-90 * Math.PI) / 180);
                    Point.applyMatrix4(boxMatrixInverse);

                    if (boxTemp.containsPoint(Point)) {
                        pointcloud.geometry.attributes.color.array[i] = 0;
                        pointcloud.geometry.attributes.color.array[i + 1] = 0.5;
                        pointcloud.geometry.attributes.color.array[i + 2] = 0;
                        green_geometry_vertices.push(Point);
                        this.prevPointsLidar.push(i, i + 1, i + 2);
                    }
                }

                pointcloud.geometry.attributes.color.needsUpdate = true;
                cube.geometry.dispose();
                cube.material.dispose();
            }
        }
    }

    async makePointCloudWhite(pointcloud) {
        let positions = pointcloud.geometry.attributes.position.array;
        console.log("pointcloud", pointcloud);
        let returnArray = [];
        for (let i = 0; i < positions.length; i += 3) {
            let x = (pointcloud.geometry.attributes.color.array[i] = 255);
            let y = (pointcloud.geometry.attributes.color.array[i + 1] = 255);
            let z = (pointcloud.geometry.attributes.color.array[i + 2] = 255);
            let v = new THREE.Vector3(x, y, z);
            let t;
            if (pointcloud.name.indexOf("Lidar") !== -1) {
                t = await this.overDistanceRedColor(
                    v,
                    "white",
                    pointcloud.name
                );
            } else {
                t = await this.overDistanceRedColor(
                    v,
                    "yellow",
                    pointcloud.name
                );
            }
            returnArray.push(t[0], t[1], t[2]);
        }
        pointcloud.geometry.attributes.color.needsUpdate = true;
    }

    async makeGreenMesh(geometry, color, box, name, layer) {
        let Float32ArrayColor = new Float32Array(color);
        let bufferGeometry = new THREE.BufferGeometry().setFromPoints(geometry);
        bufferGeometry.setAttribute(
            "color",
            new THREE.Float32BufferAttribute(Float32ArrayColor, 3)
        );

        var material = new THREE.PointsMaterial({
            size: this.pointCloudSizeLidar + 0.5,
            sizeAttenuation: false,
            vertexColors: THREE.VertexColors,
        });
        let greenMesh = new THREE.Points(bufferGeometry, material);
        const center = new THREE.Vector3(
            (box.boxModel.minVector.x + box.boxModel.maxVector.x) / 2,
            (box.boxModel.minVector.y + box.boxModel.maxVector.y) / 2,
            (box.boxModel.minVector.z + box.boxModel.maxVector.z) / 2
        );
        greenMesh.position.set(center.x, center.y, center.z);
        greenMesh.rotation.y = box.boxModel.angle;
        greenMesh.geometry.attributes.position.needsUpdate = true;
        if (name == "NS") {
            greenMesh.name = "greenMesh_Lidar_Not_Selected";
        } else {
            greenMesh.name = "greenMesh_Lidar";
            if (layer) {
                greenMesh.layers.set(layer);
            } else {
                greenMesh.layers.enable(1);
            }
        }
        this.mainController.sceneModel.scene.add(greenMesh);
    }

    async sleep(ms) {
        return new Promise((resolve) => {
            setTimeout(resolve, ms);
        });
    }
}
