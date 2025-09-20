#!/usr/bin/env python3
"""
create_subset.py
Создаёт уменьшенную версию датасета формата pitts30k (или аналогичного):
datasets/<dataset>/images/<split>/{database,queries}

Опции:
 - метод выборки: random (случайно) или grid (простая географическая дискретизация по сетке)
 - сохраняет как symlink (по умолчанию, экономит место) или копирует файлы
 - автоматически удаляет запросы без позитивов в новой базе
"""
import os
import argparse
import shutil
import numpy as np
from glob import glob
from math import floor
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-folder", required=True, help="папка с datasets")
    p.add_argument("--dataset", required=True, help="имя датасета (например pitts30k)")
    p.add_argument("--split", default="train", help="split (train/val/test)")
    p.add_argument("--splits", default=None, help="comma-separated list of splits to process, e.g. train,val,test. If set, overrides --split. Use 'all' to try train,val,test.")
    p.add_argument("--out-name", default=None, help="имя нового уменьшенного датасета (default: <dataset>_small)")
    p.add_argument("--db-size", type=int, default=5000, help="максимальное число изображений в database")
    p.add_argument("--q-size", type=int, default=None, help="максимальное число запросов (queries). None = оставить все, которые имеют позитивы")
    p.add_argument("--method", choices=["random","grid"], default="random", help="метод отбора database")
    p.add_argument("--grid-cell-size", type=float, default=30.0, help="для method=grid: размер клетки в метрах")
    p.add_argument("--use-symlinks", action="store_true", help="создавать символические ссылки вместо копирования (экономит место)")
    p.add_argument("--force", action="store_true", help="если папка назначения уже есть — удалять и перезаписывать")
    return p.parse_args()

def read_paths_and_utms(datasets_folder, dataset, split):
    base = os.path.join(datasets_folder, dataset, "images", split)
    db_folder = os.path.join(base, "database")
    q_folder = os.path.join(base, "queries")
    if not os.path.exists(db_folder) or not os.path.exists(q_folder):
        raise FileNotFoundError(f"Не найдены папки {db_folder} или {q_folder}")
    db_paths = sorted(glob(os.path.join(db_folder, "**", "*.jpg"), recursive=True))
    q_paths  = sorted(glob(os.path.join(q_folder, "**", "*.jpg"), recursive=True))
    def extract_utm(path):
        # ожидается формат path/.../@utm_easting@utm_northing@...jpg
        name = os.path.basename(path)
        parts = name.split("@")
        if len(parts) < 3:
            raise RuntimeError(f"Не удаётся извлечь UTM из имени файла: {name}")
        return float(parts[1]), float(parts[2])
    db_utms = np.array([extract_utm(p) for p in db_paths])
    q_utms  = np.array([extract_utm(p) for p in q_paths])
    return db_paths, q_paths, db_utms, q_utms

def sample_database_random(db_paths, db_utms, db_size, random_state=0):
    rng = np.random.RandomState(random_state)
    if db_size >= len(db_paths):
        idx = np.arange(len(db_paths))
    else:
        idx = rng.choice(len(db_paths), size=db_size, replace=False)
        idx = np.sort(idx)
    return idx

def sample_database_grid(db_paths, db_utms, db_size, cell_size_meter):
    # Простая дискретизация: округляем координаты по сетке cell_size_meter и оставляем по одной картинке на клетку.
    # Если после этого осталось больше, чем db_size — случайно режем.
    xs = db_utms[:,0]
    ys = db_utms[:,1]
    gx = np.floor(xs / cell_size_meter).astype(int)
    gy = np.floor(ys / cell_size_meter).astype(int)
    cells = {}
    for i, (cx, cy) in enumerate(zip(gx, gy)):
        key = (cx, cy)
        if key not in cells:
            cells[key] = i  # берем первую картинку в ячейке
    chosen = np.array(sorted(cells.values()))
    if len(chosen) > db_size:
        chosen = np.random.choice(chosen, size=db_size, replace=False)
        chosen = np.sort(chosen)
    return chosen

def ensure_positives_for_queries(selected_db_utms, q_utms, db_index_map, radius=25.0):
    # находит для каждого query есть ли в выбранной базе хотя бы один позитив в радиусе radius (метров)
    neigh = NearestNeighbors(radius=radius, n_jobs=-1)
    neigh.fit(selected_db_utms)
    # возвращает список индексов базы, попадающих в радиус для каждого запроса
    nbrs = neigh.radius_neighbors(q_utms, radius=radius, return_distance=False)
    # queries с ненулевым количеством позитивов -> keep
    good_query_mask = np.array([len(ne) > 0 for ne in nbrs])
    return good_query_mask

def make_out_dirs(datasets_folder, out_name, split, force=False):
    out_base = os.path.join(datasets_folder, out_name, "images", split)
    db_out = os.path.join(out_base, "database")
    q_out = os.path.join(out_base, "queries")
    if force and os.path.exists(os.path.join(datasets_folder, out_name)):
        shutil.rmtree(os.path.join(datasets_folder, out_name))
    os.makedirs(db_out, exist_ok=True)
    os.makedirs(q_out, exist_ok=True)
    return db_out, q_out

def link_or_copy(src, dst, use_symlink=True):
    if use_symlink:
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            return
        except Exception as e:
            logging.warning(f"Symlink failed ({e}), попробую копировать.")
    shutil.copy2(src, dst)

def run(args):
    db_paths, q_paths, db_utms, q_utms = read_paths_and_utms(args.datasets_folder, args.dataset, args.split)
    logging.info(f"Исходно: database={len(db_paths)} images, queries={len(q_paths)} images")
    if args.out_name is None:
        out_name = args.dataset + "_small"
    else:
        out_name = args.out_name

    # Выбор database
    if args.method == "random":
        db_idx = sample_database_random(db_paths, db_utms, args.db_size)
    else:
        db_idx = sample_database_grid(db_paths, db_utms, args.db_size, args.grid_cell_size)
    logging.info(f"Выбрано database: {len(db_idx)} файлов")

    selected_db_paths = [db_paths[i] for i in db_idx]
    selected_db_utms = db_utms[db_idx]

    # Для каждого query проверим, есть ли позитивы в выбранной базе (radius=25 как в коде)
    good_query_mask = ensure_positives_for_queries(selected_db_utms, q_utms, db_idx, radius=25.0)
    good_query_indices = np.where(good_query_mask)[0]
    logging.info(f"Запросов, имеющих позитивы в выбранной базе (radius=25m): {len(good_query_indices)} / {len(q_paths)}")

    # Если задан лимит q_size, уменьшить случайно до q_size
    if args.q_size is not None and len(good_query_indices) > args.q_size:
        rng = np.random.RandomState(0)
        chosen_q = rng.choice(good_query_indices, size=args.q_size, replace=False)
        chosen_q = np.sort(chosen_q)
    else:
        chosen_q = np.sort(good_query_indices)
    logging.info(f"Оставляем queries: {len(chosen_q)}")

    # Создаём output dirs
    db_out, q_out = make_out_dirs(args.datasets_folder, out_name, args.split, force=args.force)

    # Сохраняем database (symlink или копия)
    for src in selected_db_paths:
        dst = os.path.join(db_out, os.path.basename(src))
        link_or_copy(src, dst, use_symlink=args.use_symlinks)

    # Сохраняем queries (только выбранные)
    for qi in chosen_q:
        src = q_paths[qi]
        dst = os.path.join(q_out, os.path.basename(src))
        link_or_copy(src, dst, use_symlink=args.use_symlinks)

    logging.info(f"Уменьшенный датасет создан в datasets/{out_name}/images/{args.split}")
    logging.info("Готово.")

if __name__ == "__main__":
    args = parse_args()
    # Поддержка: --splits (train,val,test или all) ИЛИ --split (train/val/test/also 'all')
    if args.splits is not None:
        if args.splits.strip().lower() == "all":
            splits = ["train", "val", "test"]
        else:
            splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    else:
        # старый аргумент --split: если пользователь ошибочно написал --split all, тоже обработаем
        if args.split.strip().lower() == "all":
            splits = ["train", "val", "test"]
        else:
            splits = [args.split.strip()]

    for sp in splits:
        print(f"--- Обработка split={sp} ---")
        args.split = sp
        try:
            run(args)
        except FileNotFoundError as e:
            print(f"Warning: {e} — пропускаю split {sp}.")
        except Exception as e:
            print(f"Ошибка при обработке split={sp}: {e}")
